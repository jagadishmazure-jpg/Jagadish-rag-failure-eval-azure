// Azure AI Search + Microsoft Foundry / Azure ML hosting for the guarded retrieval service. WRITTEN, NOT RUN.
// Same resources and settings as infra/terraform (parity is checked by tests/test_iac.py).
targetScope = 'resourceGroup'

@allowed(['dev', 'prod'])
param env string
param location string = resourceGroup().location
@minLength(3)
@maxLength(10)
param prefix string = 'ragfx'
param logRetentionDays int = 30
@minValue(1)
@maxValue(3)
param searchReplicas int = 1
param tags object = {
  workload: 'rag-failure-eval'
  owner: 'ops-platform'
  data: 'synthetic'
}

var name = '${prefix}-${env}'
var flat = '${prefix}${env}'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${name}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: logRetentionDays
  }
}

resource appi 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${name}'
  location: location
  tags: tags
  kind: 'other'
  properties: {
    Application_Type: 'other'
    WorkspaceResourceId: logs.id
  }
}

resource st 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${flat}'
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowSharedKeyAccess: false
    allowBlobPublicAccess: false
    sasPolicy: { sasExpirationPeriod: '01.00:00:00', expirationAction: 'Log' }
  }
}

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${name}'
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enablePurgeProtection: true
    softDeleteRetentionInDays: 7
  }
}

resource mlw 'Microsoft.MachineLearningServices/workspaces@2025-06-01' = {
  name: 'mlw-${name}'
  location: location
  tags: tags
  kind: 'Default'
  sku: { name: 'Basic', tier: 'Basic' }
  identity: { type: 'SystemAssigned' }
  properties: {
    friendlyName: 'RAG failure eval (${env})'
    applicationInsights: appi.id
    keyVault: kv.id
    storageAccount: st.id
    systemDatastoresAuthMode: 'identity'
    hbiWorkspace: true
    publicNetworkAccess: 'Enabled'
  }
}

// Managed online endpoint: callers need an Entra ID token (no endpoint keys).
resource endpoint 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2025-06-01' = {
  parent: mlw
  name: 'ep-${name}'
  location: location
  tags: tags
  kind: 'Managed'
  identity: { type: 'SystemAssigned' }
  properties: {
    authMode: 'AADToken'
    publicNetworkAccess: 'Enabled'
    description: 'Guarded retrieval for the RAG failure eval: current documents only, hidden instructions stripped.'
  }
}

// Azure AI Search: Entra ID only (API keys disabled), managed identity, semantic ranker on the free plan.
resource search 'Microsoft.Search/searchServices@2025-05-01' = {
  name: 'srch-${name}'
  location: location
  tags: tags
  sku: { name: 'basic' }
  identity: { type: 'SystemAssigned' }
  properties: {
    replicaCount: searchReplicas
    partitionCount: 1
    disableLocalAuth: true
    publicNetworkAccess: 'enabled'
    semanticSearch: 'free'
  }
}

// Least privilege: the endpoint can read the index, nothing else (Search Index Data Reader).
resource endpointReadsIndex 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, endpoint.id, 'search-index-data-reader')
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '1407120a-92aa-4202-b7e9-c0e197c71c8f')
    principalId: endpoint.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource diagSearch 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-search'
  scope: search
  properties: {
    workspaceId: logs.id
    logs: [{ categoryGroup: 'allLogs', enabled: true }]
  }
}

resource diagMlw 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-mlw'
  scope: mlw
  properties: {
    workspaceId: logs.id
    logs: [{ categoryGroup: 'allLogs', enabled: true }]
  }
}

resource diagKv 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-kv'
  scope: kv
  properties: {
    workspaceId: logs.id
    logs: [{ category: 'AuditEvent', enabled: true }]
  }
}

output resourceGroup string = resourceGroup().name
output workspaceName string = mlw.name
output endpointName string = endpoint.name
output searchServiceName string = search.name
