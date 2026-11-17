# Offline plan tests: mocked providers, no Azure credentials, nothing created.
mock_provider "azurerm" {
  mock_data "azurerm_client_config" {
    defaults = {
      tenant_id       = "00000000-0000-0000-0000-000000000001"
      subscription_id = "00000000-0000-0000-0000-000000000002"
      object_id       = "00000000-0000-0000-0000-000000000003"
    }
  }
}

mock_provider "azapi" {}

run "dev_plan" {
  command = plan

  variables {
    env = "dev"
  }

  assert {
    condition     = azurerm_search_service.search.local_authentication_enabled == false
    error_message = "Azure AI Search must not accept API keys"
  }

  assert {
    condition     = azurerm_search_service.search.semantic_search_sku == "free"
    error_message = "semantic ranker stays on the free plan for this comparison"
  }

  assert {
    condition     = azurerm_cosmosdb_account.db.local_authentication_enabled == false
    error_message = "Cosmos DB must not accept keys"
  }

  assert {
    condition     = azurerm_cosmosdb_sql_container.chunks.throughput == 400
    error_message = "the cost model assumes 400 RU/s"
  }

  assert {
    condition     = azapi_update_resource.vectors.body.properties.resource.vectorEmbeddingPolicy.vectorEmbeddings[0].dimensions == 384
    error_message = "Cosmos vectors must match the 384-d embeddings"
  }

  assert {
    condition     = azapi_update_resource.vectors.body.properties.resource.indexingPolicy.vectorIndexes[0].type == "diskANN"
    error_message = "Cosmos vector index is diskANN"
  }

  assert {
    condition     = azurerm_postgresql_flexible_server.pg.authentication[0].password_auth_enabled == false
    error_message = "PostgreSQL password authentication must be off"
  }

  assert {
    condition     = azurerm_postgresql_flexible_server.pg.authentication[0].active_directory_auth_enabled == true
    error_message = "PostgreSQL must use Entra ID"
  }

  assert {
    condition     = azurerm_postgresql_flexible_server_configuration.vector.value == "VECTOR"
    error_message = "pgvector must be allowlisted (Azure calls the extension VECTOR)"
  }

  assert {
    condition     = azurerm_postgresql_flexible_server_active_directory_administrator.admin.principal_type == "ServicePrincipal"
    error_message = "the OIDC deploy identity is a service principal, not a user"
  }

  assert {
    condition     = azurerm_postgresql_flexible_server_firewall_rule.azure.start_ip_address == "0.0.0.0"
    error_message = "Azure services need the 0.0.0.0 firewall rule"
  }

  assert {
    condition     = azurerm_role_assignment.deployer_search.role_definition_name == "Search Index Data Contributor"
    error_message = "the deploy identity must be able to load and query the index"
  }
}

run "bad_env_rejected" {
  command = plan

  variables {
    env = "staging"
  }

  expect_failures = [var.env]
}
