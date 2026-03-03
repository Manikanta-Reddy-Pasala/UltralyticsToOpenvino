# Auto Memory

## Key Patterns

### PosClientBackend SAGA Workflow
- Workflows use `@AllArgsConstructor` for DI - just add fields for new dependencies
- `BillNumberGeneratorByBusiness` + `findAndModify` pattern for atomic sequence numbers
- Query: `businessId`, `type`, `prefix`, `deleted != true`
- `returnNew(false)` returns pre-increment value
- Existing examples: `ExpensesWorkflow.setupRecurringInvoice()`, `SalesWorkflow`, `PurchaseOrderWorkflow`

### Receipt Sequence Numbers
- Purchases use type "Tally Payment", Sales/PurchaseReturn use "Tally Receipt"
- All receipts in one transaction share the same sequence number
- Frontend receipt helper never sets appendYear/prefix for online mode
- Prefix comes from `TransactionSettings.tallyPayment.prefixSequence[0].prefix`
- `/pos/v1/user/getBillNumber` endpoint is in PosService (port 8081), NOT PosClientBackend

### Frontend AllTxnHelper.js
- `saveTxnFromPurchases` - handles AllTransaction creation for purchases
- `saveTxnFromPurchasesReturn` - same for purchase returns
- `generateReceipts()` = `generateReceiptsWithOutSequenceNumbers()` + `attachSequenceNumbersToReceipts()`
- `getLedgerForTxn()` result was dead code since `saveUpdateOrDeleteProductTxnDb` is commented out

### PosFrontend PurchasesAddStore Optimization
- `generateProductId` was async (N getBusinessData calls) → made synchronous, accepts appId param
- `getBusinessData()` reads from localStorage (fast) or queries businesslist table
- Multiple redundant `getBusinessData()` calls → cache once in saveData, pass to sub-methods
- `generateBillNumber` also called without businessData at ~line 2856 (vendor loading flow)

## Servers
- **ClawdBot VM**: 77.42.68.16 (root, SSH key auth) — runs ClawdBot at /opt/clawdbot
- **Free5GC Server**: 135.181.93.114 (root, SSH key from ClawdBot VM) — 5G SA setup, AMF with gRPC health check

## ClawdBot
- All tasks use Agent SDK by default (CLI as fallback)
- OAuth token stored in `/opt/clawdbot/.env` as `CLAUDE_CODE_OAUTH_TOKEN` (must be quoted, contains `#`)
- Deploy: `bash deploy.sh` from ClawdBot/ dir
- Repo: github.com/Manikanta-Reddy-Pasala/ClawdBot

## TallyConnector
- Reset & testing procedures in `tally-data-reset.md`
- Key field: `tallySyncedStatus: "SYNCED_FROM_TALLY"` identifies all Tally data
- Must delete `tallySyncMaster` to trigger fresh full sync
- 24 collections affected; chartOfAccounts uses filter to preserve global COA
- QA test business: `b117725333766941` (bengaluru), Tally at `77.42.45.12:9000`
- QA backend: `apiqa.oneshell.in`, user `tally-connector-qa`

## Build Commands
- PosClientBackend: `cd PosClientBackend && ./mvnw clean compile -DskipTests`
- oneshell-commons must be `mvn install` first if model classes changed
