output "resource_group" {
  description = "Resource group name."
  value       = azurerm_resource_group.rg.name
}

output "workspace_name" {
  description = "Azure ML workspace name."
  value       = azapi_resource.mlw.name
}

output "endpoint_name" {
  description = "Managed online endpoint name."
  value       = azapi_resource.endpoint.name
}

output "search_service_name" {
  description = "Azure AI Search service name."
  value       = azurerm_search_service.search.name
}
