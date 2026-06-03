import json
import os

# Procedurally generate 20 large, realistic MCP JSON schemas
# We want them to have extensive descriptions, multiple detailed properties, nested objects, and enums, simulating complex API definitions.
def generate_massive_schema(tool_id):
    # We will use realistic descriptions related to the specific tool domain
    domains = [
        ("analytics", "Enterprise Analytics Platform", "Allows deep SQL indexing, cohort building, retention metrics, and funnel analysis on massive data warehouses."),
        ("billing", "Global Billing and Subscription Gateway", "Handles invoices, payment methods, transaction reconciliation, tax compliance, and automated dunning processes."),
        ("crm", "Customer Relationship Management Suite", "Syncs sales pipelines, lead scorings, contact touchpoints, opportunity histories, and enterprise accounts."),
        ("kubernetes", "Distributed Orchestration Operator", "Monitors pod lifecycles, configures custom resource definitions, manages ingress routes, and auto-scales daemonsets."),
        ("github", "Advanced Repository Administrator", "Manages branch protections, pulls requests reviews, CI/CD action runners, hooks, issues, and org teams."),
        ("email", "Enterprise Messaging Infrastructure", "Configures SMTP servers, bounces handling, template rendering, DKIM/SPF signatures, and marketing campaigns."),
        ("database", "Distributed Relational Storage Controller", "Executes table migrations, schema introspection, backup scheduling, replica scaling, and query optimization."),
        ("cloud", "Multi-Cloud Resource Provisioner", "Manages virtual machines, virtual networks, block storage attachments, IAM policies, and VPC peering."),
        ("search", "Enterprise Semantic Indexer", "Manages vector indexing, TF-IDF configurations, cluster scaling, synonym mapping, and search query aggregations."),
        ("security", "IAM and Security Compliance Auditor", "Audits key rotations, access tokens, SSH credentials, CVE scanning, and enterprise compliance matrices.")
    ]
    
    domain_name, title, base_desc = domains[tool_id % len(domains)]
    tool_name = f"manage_{domain_name}_service_{tool_id}"
    
    properties = {}
    required = []
    
    # Build realistic JSON schema structure
    properties["client_id"] = {
        "type": "string",
        "pattern": "^[a-f0-9]{32}$",
        "description": "Unique 32-character hexadecimal identifier for the enterprise workspace instance client connection."
    }
    properties["environment"] = {
        "type": "string",
        "enum": ["production", "staging", "development", "sandbox", "testing", "integration"],
        "description": f"The target cluster deployment environment scope where the {title} commands will execute."
    }
    required.extend(["client_id", "environment"])
    
    # Generate 8 large parameters per tool to inflate the schema size realistically
    for p_idx in range(1, 9):
        p_name = f"param_field_config_{p_idx}"
        properties[p_name] = {
            "type": "object",
            "properties": {
                "enable_feature": {"type": "boolean", "default": True, "description": f"Flag to toggle feature subcomponent {p_idx} execution logic within the {title} API call context."},
                "retries": {"type": "integer", "minimum": 0, "maximum": 10, "default": 3, "description": f"Maximum retry attempts for connection recovery in component {p_idx} operations."},
                "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 60000, "default": 5000, "description": f"Network session timeout limit in milliseconds for sub-service {p_idx} calls."},
                "payload_scope": {
                    "type": "string",
                    "enum": ["minimal", "standard", "verbose", "metadata_only", "full_dump", "restricted"],
                    "description": f"Specifies the verbosity level of payload logging and reporting returned by {title} feature {p_idx}."
                },
                "nested_details": {
                    "type": "object",
                    "properties": {
                        "api_version": {"type": "string", "default": "v1.4.2", "description": f"Specific semantic API contract version targeted for nested controller {p_idx} execution."},
                        "debug_mode": {"type": "boolean", "default": False, "description": f"Enables execution tracing and verbose debugging metrics for sub-service {p_idx}."},
                        "throttling_limit": {"type": "number", "default": 250.0, "description": f"Rate limit threshold per minute allocated to client requests in feature {p_idx} context."}
                    }
                }
            },
            "required": ["enable_feature", "payload_scope"]
        }
    
    schema = {
        "name": tool_name,
        "description": f"{base_desc} Variant tool ID {tool_id}. This API integrates multiple endpoints to synchronize state across nodes.",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    return schema

def main():
    tools = []
    for i in range(1, 21):
        tools.append(generate_massive_schema(i))
        
    os.makedirs("test/schemas", exist_ok=True)
    with open("test/schemas/massive_20_tools.json", "w") as f:
        json.dump({"tools": tools}, f, indent=2)
    print("Successfully generated 20 massive MCP tool schemas in test/schemas/massive_20_tools.json")

if __name__ == "__main__":
    main()
