# Tally Data Reset Procedure

## Purpose
Reset all Tally-synced data for a business so TallyConnector can do a fresh pull.

## Key Identifier
All Tally-synced records have `tallySyncedStatus: "SYNCED_FROM_TALLY"`. Some also have `tallySync: true`, `_tallyGuid`, `_tallyAlterId`, `_tallyMasterId`.

## Collections to Reset (24 total)

### Master Data (filter by tallySyncedStatus)
- `chartOfAccounts` — delete only where `tallySyncedStatus: "SYNCED_FROM_TALLY"` (preserves global/default COA)
- `Parties`
- `businessProducts`
- `storeCategories`
- `bankAccounts`
- `costCentre` — delete ALL for business (no tallySyncedStatus, all come from Tally)
- `warehouse` — delete ALL for business (same reason)

### Voucher/Transaction Data (filter by tallySyncedStatus)
- `sales`, `purchases`, `salesReturn`, `purchasesReturn`
- `paymentIn`, `paymentOut`
- `manualJournal`, `expenses`
- `deliveryChallan`, `salesQuotation`, `saleOrder`
- `allTransactions` — delete by tallySyncedStatus AND by `isTallyOpeningBalance: true`
- `productTxn`

### Tally-Specific Collections (delete ALL for business)
- `tallyRawData` — raw master entity copies
- `tallyRawVouchers` — unmapped/invalid vouchers
- `tallyMasterSettings` — config/mapping rules
- `tallySyncMaster` — sync checkpoints (MUST delete to trigger fresh full sync)

## One-Liner Command

Replace `BUSINESS_ID` with actual business ID:

```bash
kubectl exec -n mongodb prod-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://databaseAdmin:akyFqNelEclMhlkNx06c@localhost:27017/oneshell?authSource=admin' \
  --quiet --eval '
var biz = "BUSINESS_ID";
var cols = ["Parties","businessProducts","sales","purchases","salesReturn","purchasesReturn","paymentIn","paymentOut","manualJournal","expenses","deliveryChallan","salesQuotation","saleOrder","productTxn","storeCategories","bankAccounts"];
db.chartOfAccounts.deleteMany({businessId:biz,tallySyncedStatus:"SYNCED_FROM_TALLY"});
cols.forEach(function(c){db[c].deleteMany({businessId:biz,tallySyncedStatus:"SYNCED_FROM_TALLY"})});
db.allTransactions.deleteMany({businessId:biz,tallySyncedStatus:"SYNCED_FROM_TALLY"});
db.allTransactions.deleteMany({businessId:biz,isTallyOpeningBalance:true});
["costCentre","warehouse","tallyRawData","tallyRawVouchers","tallyMasterSettings","tallySyncMaster"].forEach(function(c){db[c].deleteMany({businessId:biz})});
print("Done");
'
```

## Important Notes
- **tallySyncMaster** deletion is critical — without it, TallyConnector won't know to do a fresh pull
- **chartOfAccounts** uses `tallySyncedStatus` filter to preserve global/default COA entries
- **costCentre** and **warehouse** don't have `tallySyncedStatus`, delete all for business
- Global COA = entries without `tallySyncedStatus` field (created during business setup)
- QA DB credentials: `databaseAdmin` / `Mg#9vB@kN3wQ5z` (URL-encoded: `Mg%239vB%40kN3wQ5z`)
- QA mongos pod: `qa-cluster-mongos-0` in namespace `mongodb`
- Use `--kubeconfig /Users/manip/Documents/codeRepo/.qa-kubeconfig`

---

# Testing TallyConnector Locally

## QA Test Business
- **Business ID**: `b117725333766941`
- **Business City**: `bengaluru`
- **Tally URL**: `http://77.42.45.12:9000` (real Tally instance)

## QA Backend Credentials
- **Server URL**: `https://apiqa.oneshell.in`
- **Login URL**: `https://apiqa.oneshell.in/auth/user/token`
- **Username**: `tally-connector-qa`
- **Password**: `TallyConnectorQA@2026`

## How to Run Locally

1. **Reset Tally data** (if needed): Use the one-liner command above with QA kubeconfig

2. **Configure `application.yaml`** (or use env vars):
```yaml
tally:
  host: 77.42.45.12
  port: 9000

sync:
  pull:
    enabled: true       # Enable pull from Tally
  push:
    target-url: "https://apiqa.oneshell.in/api/v1/data/tally-ingest"
    business-id: "b117725333766941"
    business-city: "bengaluru"
    auth:
      enabled: true
      login-url: "https://apiqa.oneshell.in/auth/user/token"
      username: "tally-connector-qa"
      password: "TallyConnectorQA@2026"
  cloud-pull:
    enabled: false      # Disable cloud pull (pull-only from Tally)
```

3. **Build and run**:
```bash
cd TallyConnector
./mvnw clean package -DskipTests
java -jar target/tally-connector-*.jar
```
   Or with Maven:
```bash
cd TallyConnector
./mvnw spring-boot:run
```

4. **Select "Pull Only" mode** in the TallyConnector UI (http://localhost:8085)

5. **Verify data** in QA MongoDB after sync completes:
```bash
kubectl --kubeconfig /Users/manip/Documents/codeRepo/.qa-kubeconfig exec -n mongodb qa-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://databaseAdmin:Mg%239vB%40kN3wQ5z@localhost:27017/oneshell?authSource=admin' \
  --quiet --eval '
var biz = "b117725333766941";
print("chartOfAccounts: " + db.chartOfAccounts.countDocuments({businessId:biz}));
print("Parties: " + db.Parties.countDocuments({businessId:biz}));
print("businessProducts: " + db.businessProducts.countDocuments({businessId:biz}));
print("allTransactions: " + db.allTransactions.countDocuments({businessId:biz}));
'
```
