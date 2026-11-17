// Comparison stack for three vector stores. WRITTEN, NOT RUN.
// Same settings as infra/vectorstores/terraform. Public network access stays on because this
// stack has no VNet. Keys are disabled. PostgreSQL password authentication is off.
targetScope = 'resourceGroup'

@allowed(['dev', 'prod'])
param env string
param location string = resourceGroup().location
@minLength(3)
@maxLength(10)
param prefix string = 'ragvec'
param entraAdminName string = 'rag-failure-eval'
param tags object = {
  workload: 'rag-vector-stores'
  owner: 'ops-platform'
  data: 'synthetic'
}

var name = '${prefix}-${env}'
var flat = '${prefix}${env}'

resource search 'Microsoft.Search/searchServices@2025-05-01' = {
  name: 'srch-${name}'
  location: location
  tags: tags
  sku: { name: 'basic' }
  identity: { type: 'SystemAssigned' }
  properties: {
    replicaCount: 1
    partitionCount: 1
    disableLocalAuth: true
    publicNetworkAccess: 'enabled'
    semanticSearch: 'free'
  }
}

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: 'cosmos${flat}'
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
    locations: [
      { locationName: location, failoverPriority: 0 }
    ]
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
    minimalTlsVersion: 'Tls12'
  }
}

resource db 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = {
  parent: cosmos
  name: 'tessmoor'
  properties: {
    resource: { id: 'tessmoor' }
  }
}

// The Bicep type set for this API version lags the service: vectorEmbeddingPolicy and vectorIndexes
// are valid on a SQL container and are what the Terraform azapi update sets. The compiler warns (BCP037).
resource chunks 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: db
  name: 'chunks'
  properties: {
    resource: {
      id: 'chunks'
      partitionKey: { paths: ['/doc_id'], kind: 'Hash', version: 2 }
      vectorEmbeddingPolicy: {
        vectorEmbeddings: [
          { path: '/vector', dataType: 'float32', distanceFunction: 'cosine', dimensions: 384 }
        ]
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [{ path: '/*' }]
        excludedPaths: [{ path: '/vector/*' }, { path: '/_etag/?' }]
        vectorIndexes: [{ path: '/vector', type: 'diskANN' }]
      }
    }
    options: { throughput: 400 }
  }
}

resource pg 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: 'psql-${name}'
  location: location
  tags: tags
  sku: { name: 'Standard_B1ms', tier: 'Burstable' }
  properties: {
    version: '16'
    authConfig: {
      activeDirectoryAuth: 'Enabled'
      passwordAuth: 'Disabled'
      tenantId: subscription().tenantId
    }
    storage: { storageSizeGB: 32 }
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
    network: { publicNetworkAccess: 'Enabled' }
    availabilityZone: '1'
  }
}

resource pgdb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: pg
  name: 'tessmoor'
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

resource pgvector 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2024-08-01' = {
  parent: pg
  name: 'azure.extensions'
  properties: {
    value: 'VECTOR'
    source: 'user-override'
  }
}

resource pgadmin 'Microsoft.DBforPostgreSQL/flexibleServers/administrators@2024-08-01' = {
  parent: pg
  name: deployer().objectId
  properties: {
    principalType: 'ServicePrincipal'
    principalName: entraAdminName
    tenantId: subscription().tenantId
  }
}

// 0.0.0.0 is the Azure "allow Azure services" rule, not every address on the internet.
resource pgAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: pg
  name: 'allow-azure-services'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

resource searchWriter 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, deployer().objectId, 'search-index-data-contributor')
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8ebe5a00-799e-43f5-93ac-243d3dce84db')
    principalId: deployer().objectId
    principalType: 'ServicePrincipal'
  }
}

resource cosmosWriter 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = {
  parent: cosmos
  name: guid(cosmos.id, deployer().objectId, 'cosmos-data-contributor')
  properties: {
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    principalId: deployer().objectId
    scope: cosmos.id
  }
}

output searchServiceName string = search.name
output cosmosAccountName string = cosmos.name
output postgresServerName string = pg.name
