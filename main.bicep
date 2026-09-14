// Azure deployment for the MVP. NOT executed in this reference implementation -
// it needs a subscription. Provided as the exact shape CD expects.
//
// Component map:
//   Container Registry (ACR)   image storage, SHA-tagged, geo-replicable
//   Container Apps             serverless containers: revisions, traffic split,
//                              scale-to-zero, managed TLS. Chosen over AKS
//                              because this workload is one stateless HTTP
//                              service - a cluster would be operational cost
//                              with no benefit.
//   Key Vault                  ANTHROPIC_API_KEY, referenced by managed identity
//   Storage (Blob)             datasets, model artifacts, MLflow artifact store
//   Log Analytics + App Insights   logs, traces, metrics, alert rules
//   PostgreSQL Flexible Server the production replacement for the SQLite
//                              feedback store
param location string = resourceGroup().location
param appName string = 'novabank-assistant'
param imageTag string
param acrLoginServer string

resource logs 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${appName}-logs'
  location: location
  properties: { retentionInDays: 30, sku: { name: 'PerGB2018' } }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${appName}-insights'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web', WorkspaceResourceId: logs.id }
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${appName}-kv'
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: { type: 'SystemAssigned' }   // no credentials in config
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'http' }
      secrets: [
        { name: 'anthropic-api-key', keyVaultUrl: '${vault.properties.vaultUri}secrets/anthropic-api-key', identity: 'system' }
      ]
      registries: [ { server: acrLoginServer, identity: 'system' } ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acrLoginServer}/${appName}:${imageTag}'
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'APP_ENV', value: 'production' }
            { name: 'LLM_PROVIDER', value: 'anthropic' }
            { name: 'CONFIDENCE_THRESHOLD', value: '0.45' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: insights.properties.ConnectionString }
            { name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }
          ]
          probes: [
            { type: 'Liveness',  httpGet: { path: '/health', port: 8000 }, periodSeconds: 20 }
            { type: 'Readiness', httpGet: { path: '/ready',  port: 8000 }, periodSeconds: 10 }
          ]
        }
      ]
      scale: {
        minReplicas: 1        // not 0: cold start reloads a 23 MB artifact
        maxReplicas: 10
        rules: [ { name: 'http', http: { metadata: { concurrentRequests: '30' } } } ]
      }
    }
  }
}

output fqdn string = app.properties.configuration.ingress.fqdn
