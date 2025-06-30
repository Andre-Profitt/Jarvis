"""
OpenAPI Specification Generator for JARVIS Services
Generates comprehensive API documentation for all JARVIS tools and services
"""

import json
import yaml
from typing import Dict, List, Any
from pathlib import Path
import inspect
import importlib


class OpenAPIGenerator:
    def __init__(self):
        self.specs = {
            "openapi": "3.0.0",
            "info": {
                "title": "JARVIS AI Ecosystem API",
                "version": "1.0.0",
                "description": "Comprehensive API documentation for JARVIS unified AI system",
                "contact": {"name": "JARVIS Support", "email": "support@jarvis.ai"},
            },
            "servers": [
                {"url": "http://localhost:8000", "description": "Development server"},
                {"url": "https://api.jarvis.ai", "description": "Production server"},
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                    }
                },
            },
        }

    def generate_tool_specs(self):
        """Generate OpenAPI specs for all JARVIS tools"""

        tools = {
            "consciousness": {
                "description": "JARVIS consciousness and self-awareness system",
                "endpoints": [
                    {
                        "path": "/consciousness/state",
                        "method": "GET",
                        "summary": "Get current consciousness state",
                        "responses": {
                            "200": {
                                "description": "Current consciousness state",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/ConsciousnessState"
                                        }
                                    }
                                },
                            }
                        },
                    },
                    {
                        "path": "/consciousness/update",
                        "method": "POST",
                        "summary": "Update consciousness state",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ConsciousnessUpdate"
                                    }
                                }
                            },
                        },
                    },
                    {
                        "path": "/consciousness/think",
                        "method": "POST",
                        "summary": "Process a thought through consciousness",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "thought": {"type": "string"},
                                            "context": {"type": "object"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                ],
            },
            "scheduler": {
                "description": "Advanced task scheduling service",
                "endpoints": [
                    {
                        "path": "/scheduler/jobs",
                        "method": "GET",
                        "summary": "List all scheduled jobs",
                    },
                    {
                        "path": "/scheduler/jobs",
                        "method": "POST",
                        "summary": "Create a new scheduled job",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ScheduledJob"
                                    }
                                }
                            },
                        },
                    },
                    {
                        "path": "/scheduler/jobs/{jobId}",
                        "method": "DELETE",
                        "summary": "Cancel a scheduled job",
                        "parameters": [
                            {
                                "name": "jobId",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                    },
                ],
            },
            "service_mesh": {
                "description": "Inter-service communication hub",
                "endpoints": [
                    {
                        "path": "/mesh/services",
                        "method": "GET",
                        "summary": "List all registered services",
                    },
                    {
                        "path": "/mesh/publish",
                        "method": "POST",
                        "summary": "Publish message to service mesh",
                    },
                    {
                        "path": "/mesh/health",
                        "method": "GET",
                        "summary": "Get mesh health status",
                    },
                ],
            },
            "knowledge": {
                "description": "Semantic knowledge management",
                "endpoints": [
                    {
                        "path": "/knowledge/search",
                        "method": "POST",
                        "summary": "Semantic search in knowledge base",
                    },
                    {
                        "path": "/knowledge/store",
                        "method": "POST",
                        "summary": "Store new knowledge",
                    },
                    {
                        "path": "/knowledge/graph",
                        "method": "GET",
                        "summary": "Get knowledge graph visualization",
                    },
                ],
            },
            "project_knowledge": {
                "description": "Project-specific knowledge management",
                "endpoints": [
                    {
                        "path": "/projects/{projectId}/knowledge",
                        "method": "GET",
                        "summary": "Get project knowledge base",
                    },
                    {
                        "path": "/projects/{projectId}/search",
                        "method": "POST",
                        "summary": "Search project knowledge",
                    },
                ],
            },
            "github_copilot": {
                "description": "GitHub integration and automation",
                "endpoints": [
                    {
                        "path": "/github/repos",
                        "method": "GET",
                        "summary": "List accessible repositories",
                    },
                    {
                        "path": "/github/commit",
                        "method": "POST",
                        "summary": "Create automated commit",
                    },
                ],
            },
            "monitoring": {
                "description": "System monitoring and alerts",
                "endpoints": [
                    {
                        "path": "/monitoring/metrics",
                        "method": "GET",
                        "summary": "Get system metrics",
                    },
                    {
                        "path": "/monitoring/alerts",
                        "method": "GET",
                        "summary": "Get active alerts",
                    },
                ],
            },
            "circuit_breaker": {
                "description": "Service resilience management",
                "endpoints": [
                    {
                        "path": "/breaker/status",
                        "method": "GET",
                        "summary": "Get circuit breaker status",
                    },
                    {
                        "path": "/breaker/reset",
                        "method": "POST",
                        "summary": "Reset circuit breaker",
                    },
                ],
            },
            "feature_flags": {
                "description": "Dynamic feature management",
                "endpoints": [
                    {
                        "path": "/features",
                        "method": "GET",
                        "summary": "List all feature flags",
                    },
                    {
                        "path": "/features/{featureId}",
                        "method": "PUT",
                        "summary": "Toggle feature flag",
                    },
                ],
            },
            "ml_anomaly": {
                "description": "Machine learning anomaly detection",
                "endpoints": [
                    {
                        "path": "/anomaly/detect",
                        "method": "POST",
                        "summary": "Detect anomalies in data",
                    },
                    {
                        "path": "/anomaly/train",
                        "method": "POST",
                        "summary": "Train anomaly detection model",
                    },
                ],
            },
        }

        # Generate paths and schemas
        for tool_name, tool_info in tools.items():
            for endpoint in tool_info.get("endpoints", []):
                path = f"/api/v1{endpoint['path']}"
                method = endpoint["method"].lower()

                if path not in self.specs["paths"]:
                    self.specs["paths"][path] = {}

                self.specs["paths"][path][method] = {
                    "summary": endpoint["summary"],
                    "tags": [tool_name],
                    "security": [{"bearerAuth": []}],
                    "responses": endpoint.get(
                        "responses",
                        {
                            "200": {
                                "description": "Successful response",
                                "content": {
                                    "application/json": {"schema": {"type": "object"}}
                                },
                            },
                            "401": {"description": "Unauthorized"},
                            "500": {"description": "Internal server error"},
                        },
                    ),
                }

                if "requestBody" in endpoint:
                    self.specs["paths"][path][method]["requestBody"] = endpoint[
                        "requestBody"
                    ]

                if "parameters" in endpoint:
                    self.specs["paths"][path][method]["parameters"] = endpoint[
                        "parameters"
                    ]

        # Add component schemas
        self._add_schemas()

    def _add_schemas(self):
        """Add component schemas"""
        schemas = {
            "ConsciousnessState": {
                "type": "object",
                "properties": {
                    "awareness_level": {"type": "number", "minimum": 0, "maximum": 1},
                    "active_thoughts": {"type": "array", "items": {"type": "string"}},
                    "memory_access": {"type": "boolean"},
                    "learning_rate": {"type": "number"},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
            },
            "ConsciousnessUpdate": {
                "type": "object",
                "properties": {
                    "awareness_delta": {"type": "number"},
                    "new_thoughts": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": "object"},
                },
            },
            "ScheduledJob": {
                "type": "object",
                "required": ["name", "schedule", "task"],
                "properties": {
                    "name": {"type": "string"},
                    "schedule": {"type": "string", "description": "Cron expression"},
                    "task": {"type": "string"},
                    "params": {"type": "object"},
                    "enabled": {"type": "boolean", "default": True},
                },
            },
            "ServiceHealth": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "degraded", "down"],
                    },
                    "latency": {"type": "number"},
                    "error_rate": {"type": "number"},
                },
            },
            "KnowledgeEntry": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "metadata": {"type": "object"},
                    "embeddings": {"type": "array", "items": {"type": "number"}},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
            },
        }

        self.specs["components"]["schemas"] = schemas

    def save_specs(self, output_dir: Path):
        """Save OpenAPI specs in multiple formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(output_dir / "openapi.json", "w") as f:
            json.dump(self.specs, f, indent=2)

        # Save as YAML
        with open(output_dir / "openapi.yaml", "w") as f:
            yaml.dump(self.specs, f, default_flow_style=False)

        # Generate HTML documentation
        html_content = self._generate_html_docs()
        with open(output_dir / "api-docs.html", "w") as f:
            f.write(html_content)

    def _generate_html_docs(self) -> str:
        """Generate HTML documentation from OpenAPI specs"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>JARVIS API Documentation</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: './openapi.json',
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout"
            }});
        }}
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    generator = OpenAPIGenerator()
    generator.generate_tool_specs()
    generator.save_specs(Path("docs/api"))
    print("✅ OpenAPI specifications generated successfully!")
    print("📄 Files created:")
    print("  - docs/api/openapi.json")
    print("  - docs/api/openapi.yaml")
    print("  - docs/api/api-docs.html")
