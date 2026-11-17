"""The vector-store stack (written, not run) matches across Terraform and Bicep."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TF = (ROOT / "infra/vectorstores/terraform/main.tf").read_text()
BICEP = (ROOT / "infra/vectorstores/bicep/main.bicep").read_text()
SQL = (ROOT / "infra/pgvector/init.sql").read_text()
COMPOSE = (ROOT / "docker-compose.yml").read_text()


def test_both_stacks_describe_the_same_three_stores():
    for text in (TF, BICEP):
        assert "Microsoft.Search/searchServices" in text or "azurerm_search_service" in text
        assert "diskANN" in text
        assert "384" in text
        assert "VECTOR" in text
    assert "azurerm_search_service" in TF and "Microsoft.Search/searchServices" in BICEP
    assert "azurerm_cosmosdb_account" in TF and "Microsoft.DocumentDB/databaseAccounts" in BICEP
    assert "azurerm_postgresql_flexible_server" in TF and "Microsoft.DBforPostgreSQL/flexibleServers" in BICEP


def test_keys_and_passwords_stay_off():
    assert TF.count("local_authentication_enabled") >= 2
    assert "local_authentication_disabled" not in TF
    # Search and Cosmos DB both refuse local keys. GlobalDocumentDB contains the word DocumentDB,
    # so count the property instead of slicing the file.
    assert BICEP.count("disableLocalAuth: true") >= 2
    assert "password_auth_enabled         = false" in TF
    assert "passwordAuth: 'Disabled'" in BICEP
    assert "administrator_password" not in TF
    assert "administratorLoginPassword" not in BICEP


def test_deploy_identity_can_administer_and_query():
    for text in (TF, BICEP):
        assert "ServicePrincipal" in text
        assert "allow-azure-services" in text
        assert "0.0.0.0" in text
    assert 'role_definition_name = "Search Index Data Contributor"' in TF
    assert "8ebe5a00-799e-43f5-93ac-243d3dce84db" in BICEP
    assert "00000000-0000-0000-0000-000000000002" in TF and "00000000-0000-0000-0000-000000000002" in BICEP
    assert 'principal_type      = "User"' not in TF
    assert "principalType: 'User'" not in BICEP


def test_semantic_ranker_and_pgvector_extension():
    assert 'semantic_search_sku           = "free"' in TF
    assert "semanticSearch: 'free'" in BICEP
    assert 'value     = "VECTOR"' in TF
    assert "value: 'VECTOR'" in BICEP


def test_local_pgvector_compose_matches_the_init_script():
    assert "pgvector/pgvector:pg16" in COMPOSE
    assert "infra/pgvector/init.sql" in COMPOSE
    assert "CREATE EXTENSION IF NOT EXISTS vector" in SQL
    assert "vector(384)" in SQL
    assert "USING hnsw" in SQL
    assert "status" in SQL
