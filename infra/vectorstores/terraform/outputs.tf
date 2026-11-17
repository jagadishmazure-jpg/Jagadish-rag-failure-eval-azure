output "resource_group" {
  description = "Resource group name."
  value       = azurerm_resource_group.rg.name
}

output "search_service_name" {
  description = "Azure AI Search service name."
  value       = azurerm_search_service.search.name
}

output "cosmos_account_name" {
  description = "Cosmos DB account name."
  value       = azurerm_cosmosdb_account.db.name
}

output "postgres_server_name" {
  description = "PostgreSQL flexible server name."
  value       = azurerm_postgresql_flexible_server.pg.name
}
