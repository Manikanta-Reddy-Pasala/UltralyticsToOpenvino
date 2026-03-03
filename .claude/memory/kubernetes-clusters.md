# Kubernetes Clusters

## IMPORTANT: Always default to QA cluster unless explicitly asked for prod.

## QA Cluster (DEFAULT)
- **API Server**: https://188.245.45.41:6443
- **Kubeconfig**: `/Users/manip/Documents/codeRepo/.qa-kubeconfig`
- **Nodes**: 1 control-plane + 3 workers (k3s v1.28.15)
- **Namespaces**: default, pos, mongodb, kafka, keycloak, temporal, traefik, cert-manager
- **Always use**: `--kubeconfig /Users/manip/Documents/codeRepo/.qa-kubeconfig --insecure-skip-tls-verify`

### QA MongoDB
- **Pod**: `qa-cluster-mongos-0` in namespace `mongodb`
- **User**: `databaseAdmin`
- **Password**: `Mg#9vB@kN3wQ5z` (URL-encoded: `Mg%239vB%40kN3wQ5z`)
- **Database**: `oneshell`
- **Connection**:
```bash
kubectl --kubeconfig /Users/manip/Documents/codeRepo/.qa-kubeconfig exec -n mongodb qa-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://databaseAdmin:Mg%239vB%40kN3wQ5z@localhost:27017/oneshell?authSource=admin' \
  --quiet --eval 'QUERY_HERE'
```

### QA Services
- **API Gateway**: https://apiqa.oneshell.in
- **Port forwarding**:
```bash
kubectl --kubeconfig /Users/manip/Documents/codeRepo/.qa-kubeconfig port-forward svc/mongodbservice 8080:8080 -n default --insecure-skip-tls-verify
kubectl --kubeconfig /Users/manip/Documents/codeRepo/.qa-kubeconfig port-forward svc/posclientbackend 8090:8080 -n pos --insecure-skip-tls-verify
```

---

## Prod Cluster (USE ONLY WHEN EXPLICITLY ASKED)
- **API Server**: https://5.161.180.134:6443
- **Kubeconfig**: `/Users/manip/Documents/codeRepo/.prod-kubeconfig`
- **Always use**: `--kubeconfig /Users/manip/Documents/codeRepo/.prod-kubeconfig --insecure-skip-tls-verify`

### Prod MongoDB
- **Pod**: `prod-cluster-mongos-0` in namespace `mongodb`
- **User (app)**: `databaseAdmin` / `akyFqNelEclMhlkNx06c`
- **User (admin)**: `clusterAdmin` / `tb3GSgY6U5ZSc7CNsvf6`
- **Database**: `oneshell`
- **Connection**:
```bash
kubectl --kubeconfig /Users/manip/Documents/codeRepo/.prod-kubeconfig exec -n mongodb prod-cluster-mongos-0 --insecure-skip-tls-verify -- mongosh \
  'mongodb://databaseAdmin:akyFqNelEclMhlkNx06c@localhost:27017/oneshell?authSource=admin' \
  --quiet --eval 'QUERY_HERE'
```

### Prod Services
- **API Gateway**: https://api.oneshell.in

---

## Quick Reference

| | QA | Prod |
|---|---|---|
| Kubeconfig | `.qa-kubeconfig` | `.prod-kubeconfig` |
| API | apiqa.oneshell.in | api.oneshell.in |
| Mongos pod | qa-cluster-mongos-0 | prod-cluster-mongos-0 |
| DB password | `Mg#9vB@kN3wQ5z` | `akyFqNelEclMhlkNx06c` |
