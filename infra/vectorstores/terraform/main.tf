# Comparison stack for three vector stores. WRITTEN, NOT RUN.
# Azure AI Search (hybrid + semantic ranker), Cosmos DB for NoSQL (diskANN) and
# PostgreSQL Flexible Server with the pgvector extension allowlisted.
# The azurerm provider cannot set a Cosmos vector policy yet, so that policy is an azapi update.
# Public network access stays on for the same reason as infra/terraform: no VNet in this stack.
# Keys are disabled. PostgreSQL password authentication is off, so this file has no database password.

data "azurerm_client_config" "current" {}

locals {
  name = "${var.prefix}-${var.env}"
  flat = "${var.prefix}${var.env}"
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-${local.name}"
  location = var.location
  tags     = var.tags
}

resource "azurerm_search_service" "search" {
  name                          = "srch-${local.name}"
  location                      = azurerm_resource_group.rg.location
  resource_group_name           = azurerm_resource_group.rg.name
  sku                           = "basic"
  replica_count                 = 1
  partition_count               = 1
  local_authentication_enabled  = false
  public_network_access_enabled = true
  semantic_search_sku           = "free"
  tags                          = var.tags

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_cosmosdb_account" "db" {
  name                          = "cosmos${local.flat}"
  location                      = azurerm_resource_group.rg.location
  resource_group_name           = azurerm_resource_group.rg.name
  offer_type                    = "Standard"
  kind                          = "GlobalDocumentDB"
  local_authentication_enabled  = false
  public_network_access_enabled = true
  minimal_tls_version           = "Tls12"
  tags                          = var.tags

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = azurerm_resource_group.rg.location
    failover_priority = 0
  }
}

resource "azurerm_cosmosdb_sql_database" "db" {
  name                = "tessmoor"
  resource_group_name = azurerm_resource_group.rg.name
  account_name        = azurerm_cosmosdb_account.db.name
}

resource "azurerm_cosmosdb_sql_container" "chunks" {
  name                = "chunks"
  resource_group_name = azurerm_resource_group.rg.name
  account_name        = azurerm_cosmosdb_account.db.name
  database_name       = azurerm_cosmosdb_sql_database.db.name
  partition_key_paths = ["/doc_id"]
  throughput          = 400

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/vector/*"
    }

    excluded_path {
      path = "/_etag/?"
    }
  }
}

resource "azapi_update_resource" "vectors" {
  type        = "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15"
  resource_id = azurerm_cosmosdb_sql_container.chunks.id

  body = {
    properties = {
      resource = {
        vectorEmbeddingPolicy = {
          vectorEmbeddings = [{
            path             = "/vector"
            dataType         = "float32"
            distanceFunction = "cosine"
            dimensions       = 384
          }]
        }
        indexingPolicy = {
          indexingMode = "consistent"
          automatic    = true
          includedPaths = [{
            path = "/*"
          }]
          excludedPaths = [{
            path = "/vector/*"
            }, {
            path = "/_etag/?"
          }]
          vectorIndexes = [{
            path = "/vector"
            type = "diskANN"
          }]
        }
      }
    }
  }
}

resource "azurerm_postgresql_flexible_server" "pg" {
  name                          = "psql-${local.name}"
  resource_group_name           = azurerm_resource_group.rg.name
  location                      = azurerm_resource_group.rg.location
  version                       = "16"
  sku_name                      = "B_Standard_B1ms"
  storage_mb                    = 32768
  backup_retention_days         = 7
  geo_redundant_backup_enabled  = false
  public_network_access_enabled = true
  zone                          = "1"
  tags                          = var.tags

  authentication {
    active_directory_auth_enabled = true
    password_auth_enabled         = false
    tenant_id                     = data.azurerm_client_config.current.tenant_id
  }
}

resource "azurerm_postgresql_flexible_server_database" "db" {
  name      = "tessmoor"
  server_id = azurerm_postgresql_flexible_server.pg.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

resource "azurerm_postgresql_flexible_server_configuration" "vector" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.pg.id
  value     = "VECTOR"
}

resource "azurerm_postgresql_flexible_server_active_directory_administrator" "admin" {
  server_name         = azurerm_postgresql_flexible_server.pg.name
  resource_group_name = azurerm_resource_group.rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = data.azurerm_client_config.current.object_id
  principal_name      = "deployer"
  principal_type      = "User"
}
