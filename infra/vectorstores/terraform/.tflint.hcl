plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

plugin "azurerm" {
  enabled = true
  version = "0.32.0"
  source  = "github.com/terraform-linters/tflint-ruleset-azurerm"
}

# Written-not-run comparison stack. prevent_destroy would only block teardown.
rule "azurerm_resources_missing_prevent_destroy" {
  enabled = false
}
