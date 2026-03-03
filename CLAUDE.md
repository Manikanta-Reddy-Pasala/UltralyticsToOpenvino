# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OneShell POS monorepo - Point of Sale system with frontend apps, backend microservices, and infrastructure. Supports two deployment modes: Docker/local (offline-capable) and Web/Cloud (Kubernetes).

## Build Commands

### Java (Spring Boot)
```bash
./mvnw clean package -DskipTests        # Build
./mvnw spring-boot:run                  # Run locally
./mvnw test                             # Run all tests
./mvnw test -Dtest=ClassName            # Run single test class
./mvnw test -Dtest=ClassName#methodName # Run single test method
./mvnw -Pnative native:compile -DskipTests  # GraalVM native (PosClientBackend)
```

### Shared Library (oneshell-commons)
```bash
cd oneshell-commons
./mvnw clean install -DskipTests        # Install to local Maven repo (required before building dependent services)
```

### Frontend
```bash
# PosFrontend (React 16.14 + Electron)
cd PosFrontend && npm install && npm run start   # Dev server
npm run build-electron                           # Production build

# PosAdmin (React 18.3 + Vite)
cd PosAdmin && yarn install && yarn dev          # Dev server (port 5174)
yarn build                                       # Production build
yarn lint                                        # Run ESLint
```

### Python (PosPythonBackend)
```bash
pip install -r requirements.txt && python main.py  # Port 5100
python -m pytest                                    # Run tests
```

## Architecture

### Critical Rules
1. **MongoDbService is MANDATORY** - Never query MongoDB directly from any service
2. **PosClientBackend changes affect TWO deployments** - Docker AND K8s `pos` namespace
3. **Sync flow**: Docker PosClientBackend → NATS (`business.push.request`) → PosServerBackend → MongoDbService → MongoDB

### Deployment Modes
1. **Docker/Local**: PosClientBackend in Docker, syncs to PosServerBackend via REST/NATS (offline-capable)
2. **Web/Cloud**: PosClientBackend in K8s `pos` namespace, direct MongoDB access

### Core Services

| Service | Tech | Port | Purpose |
|---------|------|------|---------|
| PosClientBackend | Spring Boot 3.3/Java 21/WebFlux | 8090 | Main client API, GraphQL. Docker + K8s |
| PosServerBackend | Spring Boot 3.3/Java 21/WebFlux | 8091 | Cloud backend, receives Docker sync, change streams |
| MongoDbService | Spring Boot 3.1/Java 17 | 8080 | **MANDATORY** MongoDB gateway |
| GatewayService | Spring Boot 2.7/Java 17 | 8080 | API Gateway, JWT auth |
| BusinessService | Spring Boot 2.7/Java 21 | 8080 | Business logic for PosAdmin/mobile |
| PosService | Spring Boot/Java 21 | 8081 | POS operations |
| Scheduler | Spring Boot 3.2/Java 21 | 8080 | Recurring events, invoices |
| QuartzScheduler | Spring Boot 3.2/Java 21 | 8080 | Stock/balance corrections, batch jobs |

### Supporting Services
- **PosPythonBackend** (Flask:5100): OCR, bank parsing, AI Assistant (Ollama)
- **EmailService/NotificationService/WhatsappApiService**: Communications
- **CacheLayer**: Redis/Dragonfly caching
- **PosDataSyncService**: Tally bidirectional sync

### Shared Library (oneshell-commons)

| Module | Purpose |
|--------|---------|
| `oneshell-commons-model` | DAO objects, domain models (ProductTxnDao, SalesQuotationDao, BusinessProfile, etc.) |
| `oneshell-commons-mongodb` | Reactive MongoDB utilities |
| `oneshell-commons-nats` | NATS JetStream client (v2.20.2) |

Models are in package: `com.oneshell.commons.server.model.v1`

### Data Sync Architecture (Detailed)

The sync system has two main flows: **Push Sync** (PosClientBackend → PosServerBackend) and **Change Stream Sync** (PosServerBackend → child businesses).

#### End-to-End Sync Flow

```
PosClientBackend (Docker)
  │ publishToRemoteServer()
  ▼
Local NATS JetStream (stream: localMessageStream, subject: pendingToSync)
  │ MessageRetryService polls every 30s, batch of 50
  ▼
HTTP POST → PosServerBackend /v1/pos-server/api/nats/publish
  │ ClientSyncPushRequestConsumer (NATS subject: business.push.request, queue: pos-server-backend-queue)
  ▼
DataConversionService.processIncomingData() → MongoDbService → MongoDB
  │ Change stream detects insert/update/delete
  ▼
MongoDBTransactionsChangeStreamListener (leader-elected, 1 pod only)
  │ Applies filters (e.g., chartOfAccounts Sundry Debtor/Creditor skip)
  │ Checks _syncSource to prevent loops
  ▼
JetStream publish (stream: changestream-events, subject: changestream.events.{collection})
  │ ChangeStreamEventConsumer polls every 2s, batch of 50
  ▼
TransactionSyncRulesServiceImpl.applyRulesForBusiness()
  │ Fetches rules by fromBusinessId, applies transforms
  ▼
MongoDbService HTTP → MongoDB (synced doc with ID: {originalId}_{toBusinessId}, _syncSource marker)
```

---

#### 1. PosClientBackend Push Sync

**Key Files:**
- `PosClientBackend/src/main/java/com/pos/backend/dataSync/PosServerBackendService.java`
- `PosClientBackend/src/main/java/com/pos/backend/dataSync/MessageRetryService.java`
- `PosClientBackend/src/main/java/com/pos/backend/dataSync/DataPullRequestClient.java`

**Push Flow (fire-and-forget):**
1. Service calls `publishToRemoteServer(payload, tableName, action)` after saving locally
2. Validates params, checks feature toggle (`serverSync.push.enabled`)
3. Publishes to local NATS JetStream (stream: `localMessageStream`, subject: `pendingToSync`)
4. Returns immediately with `NATS_QUEUED_SEQ_{seqNo}` - caller never blocks

**NATS Message Headers:** `tableName`, `action` (Save/Update/Delete), `traceId` (UUID)

**MessageRetryService (background retry loop):**
- Polls `pendingToSync` every 30 seconds, batch size 50, max 10 concurrent
- Durable consumer: `message-retry-durable` (survives restarts)
- Sends HTTP POST to PosServerBackend `/v1/pos-server/api/nats/publish` (10s timeout)
- On success (HTTP 200 + SUCCESS): ACK message (removed from queue)
- On failure: NAK with 5-10s delay for retry

**Feature Toggles** (`application.yaml`):
```yaml
feature:
  serverSync:
    mode: AUTO        # AUTO, PUSH, or PULL
    push:
      enabled: true
    pull:
      enabled: true
```

#### 2. PosClientBackend Pull Sync

**Key File:** `PosClientBackend/src/main/java/com/pos/backend/dataSync/DataPullRequestClient.java`

**Trigger:** Scheduler (every 5 minutes default), or manual via `/v1/sync/force`

**Pull Flow:**
1. Gets businesses (selected businesses processed first = higher priority)
2. For each business, gets eligible tables (not currently syncing, due for sync based on interval)
3. Sorts by priority: settings tables (priority 10) first, then by ascending priority
4. Processes up to `maxTablesPerBusiness` (default 2) concurrently
5. Sends HTTP POST to `/v1/pos-server/api/nats/pull-request` with `{businessId, tableName, pageNumber, pageSize, updatedAt}`
6. Paginates: if `response.size() == pageSize`, fetches next page
7. Saves records locally via `ServerToClientDataProcessingService.processInComingData()`
8. Updates `lastSyncedTimestamp` for next incremental pull

**Conflict Resolution:** Last-write-wins based on `updatedAt` timestamp. Server data overwrites local.

**Table Priority Tiers:**

| Priority | Tables | Purpose |
|----------|--------|---------|
| 10 | All *Settings tables | Configuration - blocks UI |
| 20 | chartOfAccounts | Accounting setup |
| 30 | productTxn, allTransactions, sales | Large critical tables |
| 35 | salesReturn, payments, purchases | Medium priority |
| 40 | businessProducts, Parties, expenses | Large/important |
| 45 | deliveryChallan, salesQuotation, saleOrder | Orders |
| 50+ | Other tables | Low priority |

**Sync Config** (`PosClientBackend/src/main/resources/application.yaml`):
```yaml
data:
  sync:
    defaultInterval: 300000          # 5 minutes
    defaultPageSize: 300
    maxConcurrentServerRequests: 3
    maxTablesPerBusiness: 2
```

---

#### 3. PosServerBackend - NATS Consumer (Receiving Push Sync)

**Key File:** `PosServerBackend/src/main/java/com/pos/backend/nats/consumer/ClientSyncPushRequestConsumer.java`

**Configuration:**
- NATS subject: `business.push.request`
- Queue group: `pos-server-backend-queue` (only 1 replica processes each message)
- Processing timeout: 120 seconds (blocking)
- Lock: `lock:posserverbackend:sync:{businessId}` (10 min timeout, 30s wait)

**Processing Flow:**
1. Extract headers: `tableName`, `action`, `businessId`, `deviceId`
2. Acquire distributed lock per businessId (prevents concurrent syncs for same business)
3. Deserialize via `DataConversionService.getDeserializedMessage()` (supports 80+ table types)
4. Process via `DataConversionService.processIncomingData()` (saves to MongoDB via MongoDbService)
5. Send response via NATS replyTo: `SUCCESS:Processed message successfully for table: {tableName}`
6. Release lock in finally block

---

#### 4. PosServerBackend - Change Stream Listener

**Key File:** `PosServerBackend/src/main/java/com/pos/backend/eventListner/MongoDBTransactionsChangeStreamListener.java`

**Monitored Collections:** `storeCategories`, `businessProducts`, `sales`, `saleOrder`, `salesQuotation`, `Parties`, `chartOfAccounts`, `industry`, `leadSource`, `leadStatus`

**Leader Election (Redisson):**
- Lock key: `lock:posserverbackend:change-stream-leader`
- Check interval: 60 seconds
- Only the leader pod starts change stream listeners
- Watchdog auto-renews lock every 30 seconds

**Health Checks:**
- Interval: 120 seconds
- Detects stuck streams (no activity for 600,000ms / 10 minutes)
- Detects disposed subscriptions
- Auto-restarts unhealthy streams

**Resume Tokens:**
- Stored in Redis: `changestream:resume-token:{collection}` (TTL: 7 days)
- Saved BEFORE processing (at-least-once delivery guarantee)
- On `ChangeStreamHistoryLost` error: clears token, restarts from current position

**Change Event Processing:**
1. **Skip synced docs:** Documents with `_syncSource` field are skipped (prevents infinite loops)
2. **Validate fields:** Skip if `businessId` is null/empty
3. **Apply filters:**
   - **chartOfAccounts filter:** Skip last-level accounts under Sundry Debtors (`parentId: 6c9d9954-c434-4644-922c-d673899b3978`) or Sundry Creditors (`parentId: 48c5cc6b-cd94-48ca-bd1d-f7d8012fa05b`) when `isLastLevel: true` - these are individual customer/vendor accounts that should not sync to child businesses
4. **Route to processing:**
   - If JetStream enabled (default): Publish to `changestream.events.{collection}` for async processing
   - If JetStream disabled (fallback): Process inline via `syncService.applyRulesForBusiness()`
   - If JetStream publish fails: Falls back to inline processing

**Delete Event Processing:**
- Synced document IDs follow pattern: `{originalId}_{childBusinessId}`
- Extracts childBusinessId from ID suffix
- Calls `syncService.applyDeleteSync()` to remove synced copies

---

#### 5. PosServerBackend - JetStream Consumer

**Key File:** `PosServerBackend/src/main/java/com/pos/backend/nats/consumer/ChangeStreamEventConsumer.java`

**Configuration:**
- Stream: `changestream-events` (WorkQueue retention, 7-day max age)
- Consumer: `changestream-processor` (durable)
- Batch size: 50 messages
- Poll timeout: 5 seconds
- Poll interval: 2 seconds
- Max delivery attempts: 3
- ACK wait: 60 seconds

**Processing:**
- Polls JetStream every 2 seconds for up to 50 messages
- For insert/update: calls `syncService.applyRulesForBusiness(collection, docJson)`
- For delete: calls `syncService.applyDeleteSync()` (only for `businessProducts`)
- On success: `msg.ack()`

**Retry & DLQ:**
- Backoff: 5s (1st), 30s (2nd), 120s (3rd)
- After max retries or non-retryable error: `msg.term()`, publish to DLQ stream (`changestream-dlq`, 30-day retention)
- Errors logged to MongoDB `changeStreamEventErrors` collection

**Key File:** `PosServerBackend/src/main/java/com/pos/backend/nats/publisher/ChangeStreamEventPublisher.java`
- `publishEventSync()`: Blocking publish used by change stream listener
- `publishToDeadLetterQueue()`: DLQ with error context, timestamp, attempt count
- Subject format: `changestream.events.{collection}` / `changestream.dlq.{collection}`

---

#### 6. PosServerBackend - Transaction Sync Rules

**Key File:** `PosServerBackend/src/main/java/com/pos/backend/service/Impl/TransactionSyncRulesServiceImpl.java`

**Sync Rule Model** (collection: `transactionSyncRules`):
```json
{
  "_id": "rule-id",
  "transactionType": "chartOfAccounts",
  "fromBusinessId": "b117695104178401",
  "toBusinessIds": ["b117696659790651", "b117695104665631"],
  "active": true,
  "syncMode": "ALL",
  "rules": [
    { "field": "fieldName", "action": "set", "value": "newValue" },
    { "field": "price", "action": "multiply", "value": 1.1 }
  ],
  "lastAllSyncTimestamp": 1769965508152
}
```

**applyRulesForBusiness(txnType, docJson):**
1. Extract `businessId` from document
2. Fetch active rules where `fromBusinessId` matches document's businessId
3. For each rule, for each `toBusinessId`:
   - Transform document ID: `{originalId}_{toBusinessId}`
   - Set `businessId` to target business
   - Apply transform rules (set, set_null, multiply, sum, subtract, array field transforms)
   - Add `_syncSource` marker: `{originalId, fromBusinessId, syncedAt}`
   - Check for existing document (skip if already synced = duplicate prevention)
   - Save via MongoDbService HTTP call

**applyDeleteSync(collection, documentId, businessId):**
1. Fetch active rules where `fromBusinessId` matches
2. For each target business, delete document with ID: `{documentId}_{targetBusinessId}`

**Full Sync (syncMode: ALL):**
- Triggered when `lastAllSyncTimestamp` is null/0 (new rule or reset)
- Paginated: 300 documents per page
- Processes all documents from `fromBusinessId` and syncs to all `toBusinessIds`
- Updates `lastAllSyncTimestamp` after completion

**Transform Rule Actions:** `set`, `set_null`, `multiply`, `sum`, `subtract`, array transforms (`field[*].subfield`)

**Sync Loop Prevention:**
- Synced documents get `_syncSource` field added
- Change stream listener skips documents with `_syncSource` (line 566-572 in listener)
- This prevents: Parent→Child sync triggering Child→Parent→Child infinite loop

---

#### 7. TransactionSyncRules Listener

**Key File:** `PosServerBackend/src/main/java/com/pos/backend/eventListner/TransactionSyncRulesListener.java`

- Monitors `transactionSyncRules` collection via change stream
- Leader-elected (lock: `lock:posserverbackend:sync-rules-listener-leader`)
- When a new rule is created/updated with `syncMode: ALL` and `lastAllSyncTimestamp` is null/0, triggers full sync
- Retries up to 5 times on failure

---

#### 8. Error Handling

**Key File:** `PosServerBackend/src/main/java/com/pos/backend/service/url/ChangeStreamErrorService.java`

**MongoDB Collection:** `changeStreamEventErrors`
- Fields: businessId, collection, documentId, operationType, errorType, retryable, retryCount, contextData, createdAt, resolved
- Query unresolved errors: `db.changeStreamEventErrors.find({resolved: false}).sort({createdAt: -1})`

**NonRetryableException:** Used to immediately send to DLQ without retrying (e.g., null document JSON, invalid data)

## Kubernetes Operations

**IMPORTANT**: Always use `--insecure-skip-tls-verify`

```bash
kubectl get pods -n pos --insecure-skip-tls-verify
kubectl logs -f deployment/posclientbackend -n pos --insecure-skip-tls-verify
kubectl rollout restart deployment/posclientbackend -n pos --insecure-skip-tls-verify
```

### Namespaces
- `default`: MongoDbService, GatewayService, BusinessService, Scheduler, QuartzScheduler, EmailService
- `pos`: PosClientBackend, PosPythonBackend, NATS, Ollama
- `mongodb`: Percona MongoDB Operator, prod-cluster

### Port Forwarding
```bash
kubectl port-forward svc/mongodbservice 8080:8080 -n default --insecure-skip-tls-verify
kubectl port-forward svc/posclientbackend 8090:8080 -n pos --insecure-skip-tls-verify
kubectl port-forward svc/pospythonbackend 5100:5100 -n pos --insecure-skip-tls-verify
```

## MongoDB (Percona Operator)

Sharded cluster in `mongodb` namespace: 3 Config Servers, 2 Mongos, 3 RS Members

### Connection
```bash
# databaseAdmin (app access)
kubectl exec -n mongodb prod-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://databaseAdmin:akyFqNelEclMhlkNx06c@localhost:27017/oneshell?authSource=admin' \
  --quiet --eval 'db.stats()'

# clusterAdmin (admin only)
kubectl exec -n mongodb prod-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://clusterAdmin:tb3GSgY6U5ZSc7CNsvf6@localhost:27017/admin?authSource=admin' \
  --quiet --eval 'db.adminCommand({listDatabases: 1})'
```

### Common Commands
```bash
# Kill sessions (fixes TooManyLogicalSessions)
kubectl exec -n mongodb prod-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://clusterAdmin:tb3GSgY6U5ZSc7CNsvf6@localhost:27017/admin?authSource=admin' \
  --quiet --eval 'db.adminCommand({killAllSessions: []})'

# Check sync errors
kubectl exec -n mongodb prod-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://databaseAdmin:akyFqNelEclMhlkNx06c@localhost:27017/oneshell?authSource=admin' \
  --quiet --eval 'db.changeStreamEventErrors.find({resolved: false}).sort({createdAt: -1}).limit(10).forEach(e => print(e.collection + " | " + e.errorType))'
```

### Key Collections
`productTxn`, `allTransactions`, `sales`, `sagaTransactions`, `businessProducts`, `Parties`, `employees`

### Config Files
- Cluster: `SetupRelated/cluster_setup/mongoDb/mongodb-6.0-prod.yaml`
- Maintenance: `SetupRelated/cluster_setup/mongoDb/mongodb-maintenance-cronjobs.yaml`

## CI/CD (Tekton)

### Release Process
- **QA**: Push to `master` → auto-deploys with commit SHA tag
- **Production**: `git tag v1.x.x && git push origin v1.x.x`

### Java Version Mapping
- **Java 21**: PosServerBackend, PosService, BusinessService, CacheLayer, Scheduler, QuartzScheduler, PosClientBackend
- **Java 17**: MongoDbService, WhatsappApiService, GstApiService, NotificationService, EmailService, GatewayService

### GitOps
- Repo: `github.com/OneShellSolutions/gitops-repo`
- QA: `qa-apps/{service}/deployment.yaml`
- Prod: `apps/{service}/deployment.yaml`

## Agent Workflow

### Java Projects
```bash
mvn clean compile && mvn test && mvn clean package -DskipTests
git add . && git commit -m "feat: description" && git push origin master
# Deploy (only if asked): git tag v1.x.x && git push origin v1.x.x
```

### Python Projects
```bash
pip install -r requirements.txt && python -m py_compile main.py
git add . && git commit -m "feat: description" && git push origin master
```

### React/Node Projects
```bash
npm install && npm run lint && npm run build
git add . && git commit -m "feat: description" && git push origin master
```

### Rules
1. **Never deploy automatically** - only when user explicitly asks
2. **Always compile/test before committing**
3. **No sensitive data** in commits

## Key Components

### PosServerBackend
- **Change Stream Listener**: `MongoDBTransactionsChangeStreamListener` monitors 10 collections
- **NATS Consumer**: `business.push.request` subject, `pos-server-backend-queue` group
- **JetStream Consumer**: `ChangeStreamEventConsumer` polls `changestream-events` stream
- **JetStream Publisher**: `ChangeStreamEventPublisher` publishes to `changestream.events.{collection}`
- **Sync Rules Engine**: `TransactionSyncRulesServiceImpl` - rule-based multi-business sync
- **Sync Rules Listener**: `TransactionSyncRulesListener` - watches rule changes, triggers full sync
- **Lock Manager**: `LockManager` - Redisson distributed locks for sync and publish
- **Data Conversion**: `DataConversionServiceImpl` - deserializes/processes 80+ table types
- **Error Logging**: `ChangeStreamErrorService` → `changeStreamEventErrors` collection
- **NATS Config**: `NatsConfig` - JetStream streams, consumers, NATS connection

### PosClientBackend
- **Push Sync**: `PosServerBackendService` in `dataSync/` - publishes to local NATS, fire-and-forget
- **Retry Service**: `MessageRetryService` in `dataSync/` - polls local NATS, HTTP POST to server
- **Pull Sync**: `DataPullRequestClient` in `dataSync/` - incremental pull by updatedAt
- **Feature Toggles**: `FeatureToggleService` - serverSync.push.enabled, serverSync.pull.enabled
- **Sync Config**: `SyncConfig` - intervals, page sizes, table priorities
- **GraphQL API**: Available at `/pos/api/query`
- **REST APIs**: Client sync/pull operations

### Redis Keys
- `lock:posserverbackend:sync:{businessId}` - Sync lock (10 min timeout)
- `lock:posserverbackend:publish:sync:{businessId}` - Publish lock
- `lock:posserverbackend:change-stream-leader` - Change stream leader election
- `lock:posserverbackend:sync-rules-listener-leader` - Sync rules listener leader
- `changestream:resume-token:{collection}` - Resume tokens (7-day TTL)
- `shedlock:{taskName}` - Distributed scheduling

### NATS Subjects & Streams
- `business.push.request` - PosClientBackend → PosServerBackend push sync (queue: `pos-server-backend-queue`)
- `pendingToSync` - PosClientBackend local queue (stream: `localMessageStream`, consumer: `message-retry-durable`)
- `changestream.events.{collection}` - Change stream events (stream: `changestream-events`, consumer: `changestream-processor`)
- `changestream.dlq.{collection}` - Dead letter queue (stream: `changestream-dlq`, 30-day retention)

### Key Timeouts & Limits

| Parameter | Value |
|-----------|-------|
| Push sync HTTP timeout | 10 seconds |
| NATS processing timeout | 120 seconds |
| Sync lock timeout | 10 minutes |
| JetStream poll interval | 2 seconds |
| JetStream batch size | 50 messages |
| JetStream max delivery | 3 attempts |
| JetStream retry backoff | 5s, 30s, 120s |
| JetStream ACK wait | 60 seconds |
| Change stream health check | 120 seconds |
| Change stream inactivity timeout | 10 minutes |
| Pull sync default interval | 5 minutes |
| Pull sync default page size | 300 records |
| MessageRetryService poll interval | 30 seconds |
| MessageRetryService batch size | 50 messages |

## Key File Locations

| Component | Path |
|-----------|------|
| **Shared Models** | `oneshell-commons/oneshell-commons-model/src/main/java/com/oneshell/commons/server/model/v1/` |
| **PosServerBackend - Change Stream Listener** | `PosServerBackend/src/main/java/com/pos/backend/eventListner/MongoDBTransactionsChangeStreamListener.java` |
| **PosServerBackend - JetStream Consumer** | `PosServerBackend/src/main/java/com/pos/backend/nats/consumer/ChangeStreamEventConsumer.java` |
| **PosServerBackend - JetStream Publisher** | `PosServerBackend/src/main/java/com/pos/backend/nats/publisher/ChangeStreamEventPublisher.java` |
| **PosServerBackend - NATS Push Consumer** | `PosServerBackend/src/main/java/com/pos/backend/nats/consumer/ClientSyncPushRequestConsumer.java` |
| **PosServerBackend - Sync Rules Service** | `PosServerBackend/src/main/java/com/pos/backend/service/Impl/TransactionSyncRulesServiceImpl.java` |
| **PosServerBackend - Sync Rules Listener** | `PosServerBackend/src/main/java/com/pos/backend/eventListner/TransactionSyncRulesListener.java` |
| **PosServerBackend - Data Conversion** | `PosServerBackend/src/main/java/com/pos/backend/service/Impl/DataConversionServiceImpl.java` |
| **PosServerBackend - Error Service** | `PosServerBackend/src/main/java/com/pos/backend/service/url/ChangeStreamErrorService.java` |
| **PosServerBackend - Lock Manager** | `PosServerBackend/src/main/java/com/pos/backend/lock/LockManager.java` |
| **PosServerBackend - NATS Config** | `PosServerBackend/src/main/java/com/pos/backend/config/NatsConfig.java` |
| **PosServerBackend - Event Message Model** | `PosServerBackend/src/main/java/com/pos/backend/model/ChangeStreamEventMessage.java` |
| **PosClientBackend - Push Sync Service** | `PosClientBackend/src/main/java/com/pos/backend/dataSync/PosServerBackendService.java` |
| **PosClientBackend - Message Retry** | `PosClientBackend/src/main/java/com/pos/backend/dataSync/MessageRetryService.java` |
| **PosClientBackend - Pull Sync Client** | `PosClientBackend/src/main/java/com/pos/backend/dataSync/DataPullRequestClient.java` |
| **PosClientBackend - Sync Config** | `PosClientBackend/src/main/resources/application.yaml` (data.sync section) |
| **K8s Deployments** | `SetupRelated/cluster_setup/argo-cd/apps/` |
| **MongoDB Setup** | `SetupRelated/cluster_setup/mongoDb/mongodb-6.0-prod.yaml` |

## Service-Specific Docs
- `PosPythonBackend/CLAUDE.md` - OCR, bank parsing, AI Assistant
- `PosDataSyncService/CLAUDE.md` - Tally bidirectional integration
