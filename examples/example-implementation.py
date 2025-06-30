# example_ecommerce_project.py
"""
Practical example: Using the Autonomous Project Engine to build an e-commerce platform
This demonstrates real-world usage with specific agents and detailed configuration
"""

import asyncio
from datetime import datetime, timedelta
from autonomous_project_engine import (
    AutonomousProjectEngine,
    BaseAgent,
    AgentCapability,
    AgentRole,
    Task,
    ProjectContext,
)


class DatabaseArchitectAgent(BaseAgent):
    """Specialized agent for database design and optimization"""

    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                "schema_design", 0.9, ["postgresql", "mongodb"], ["relational", "nosql"]
            ),
            AgentCapability(
                "query_optimization", 0.85, ["explain", "indexing"], ["performance"]
            ),
            AgentCapability(
                "data_modeling",
                0.88,
                ["normalization", "denormalization"],
                ["scalability"],
            ),
        ]
        super().__init__(agent_id, AgentRole.EXECUTOR, capabilities)

    async def execute(self, task: Task, context: ProjectContext) -> Dict[str, Any]:
        """Design optimal database schema for e-commerce"""

        # Analyze requirements for database design
        schema_design = {
            "tables": {
                "users": {
                    "columns": [
                        "id",
                        "email",
                        "password_hash",
                        "created_at",
                        "last_login",
                    ],
                    "indexes": ["email", "created_at"],
                    "partitioning": "by_year(created_at)",
                },
                "products": {
                    "columns": [
                        "id",
                        "name",
                        "description",
                        "price",
                        "inventory",
                        "category_id",
                    ],
                    "indexes": ["category_id", "price", "name"],
                    "full_text_search": ["name", "description"],
                },
                "orders": {
                    "columns": ["id", "user_id", "total", "status", "created_at"],
                    "indexes": ["user_id", "status", "created_at"],
                    "partitioning": "by_month(created_at)",
                },
                "order_items": {
                    "columns": ["id", "order_id", "product_id", "quantity", "price"],
                    "indexes": ["order_id", "product_id"],
                },
            },
            "optimization_strategies": [
                "Use read replicas for product catalog queries",
                "Implement caching layer for frequently accessed data",
                "Use materialized views for analytics queries",
            ],
            "scaling_plan": {
                "sharding_strategy": "by_user_id for orders",
                "replication": "master-slave with 2 read replicas",
                "backup": "daily incremental, weekly full",
            },
        }

        return {
            "task_id": task.id,
            "schema": schema_design,
            "estimated_performance": {
                "read_qps": 50000,
                "write_qps": 5000,
                "storage_growth": "100GB/month",
            },
            "recommendations": [
                "Consider NoSQL for product catalog if variety increases",
                "Implement event sourcing for order history",
                "Use time-series database for analytics",
            ],
        }

    async def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from database performance metrics"""
        self.memory.append(experience)

        if "query_performance" in experience:
            # Adjust optimization strategies based on actual performance
            avg_query_time = experience["query_performance"]["avg_time"]
            if avg_query_time > 100:  # ms
                self.capabilities[1].skill_level *= 0.95  # Reduce confidence
            else:
                self.capabilities[1].skill_level = min(
                    0.95, self.capabilities[1].skill_level * 1.02
                )


class SecurityAuditAgent(BaseAgent):
    """Specialized agent for security assessment and implementation"""

    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                "vulnerability_scanning", 0.92, ["owasp", "burp"], ["web_security"]
            ),
            AgentCapability(
                "authentication_design", 0.88, ["oauth", "jwt"], ["identity"]
            ),
            AgentCapability(
                "encryption_implementation", 0.9, ["aes", "rsa"], ["data_protection"]
            ),
            AgentCapability(
                "compliance_verification", 0.85, ["pci", "gdpr"], ["regulations"]
            ),
        ]
        super().__init__(agent_id, AgentRole.QA_TESTER, capabilities)

    async def execute(self, task: Task, context: ProjectContext) -> Dict[str, Any]:
        """Perform comprehensive security audit and implementation"""

        security_assessment = {
            "vulnerabilities_found": [],
            "security_measures": {
                "authentication": {
                    "method": "JWT with refresh tokens",
                    "mfa": "TOTP-based 2FA",
                    "session_management": "Redis-based with 30min timeout",
                },
                "authorization": {
                    "model": "RBAC with granular permissions",
                    "api_security": "Rate limiting + API keys",
                },
                "data_protection": {
                    "encryption_at_rest": "AES-256",
                    "encryption_in_transit": "TLS 1.3",
                    "sensitive_data": "Tokenization for payment info",
                },
                "infrastructure": {
                    "waf": "CloudFlare WAF enabled",
                    "ddos_protection": "Rate limiting + CDN",
                    "monitoring": "Real-time threat detection",
                },
            },
            "compliance_status": {
                "pci_dss": "Compliant with SAQ-D",
                "gdpr": "Privacy by design implemented",
                "ccpa": "User data controls in place",
            },
            "security_score": 0.88,
        }

        # Simulate vulnerability scanning
        vulnerabilities = await self._scan_for_vulnerabilities(context)
        security_assessment["vulnerabilities_found"] = vulnerabilities

        return {
            "task_id": task.id,
            "assessment": security_assessment,
            "critical_actions": [
                "Implement CSP headers",
                "Enable HSTS",
                "Set up security monitoring alerts",
            ],
            "estimated_implementation_time": "2 weeks",
        }

    async def _scan_for_vulnerabilities(self, context: ProjectContext) -> List[Dict]:
        """Simulate vulnerability scanning"""
        # In real implementation, this would use actual security tools
        return [
            {
                "type": "Missing Security Headers",
                "severity": "Medium",
                "fix": "Add X-Frame-Options, X-Content-Type-Options headers",
            }
        ]

    async def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from security incidents and audits"""
        self.memory.append(experience)

        if "security_incidents" in experience:
            # Update security strategies based on incidents
            for incident in experience["security_incidents"]:
                if incident["type"] not in self.known_threats:
                    self.known_threats.append(incident["type"])


class PerformanceOptimizationAgent(BaseAgent):
    """Agent specialized in performance optimization and scalability"""

    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                "load_testing", 0.87, ["jmeter", "locust"], ["stress_testing"]
            ),
            AgentCapability(
                "caching_strategy", 0.9, ["redis", "memcached"], ["optimization"]
            ),
            AgentCapability(
                "cdn_configuration", 0.85, ["cloudflare", "fastly"], ["distribution"]
            ),
            AgentCapability(
                "code_profiling", 0.88, ["profilers", "apm"], ["bottlenecks"]
            ),
        ]
        super().__init__(agent_id, AgentRole.EXECUTOR, capabilities)

    async def execute(self, task: Task, context: ProjectContext) -> Dict[str, Any]:
        """Optimize system performance for e-commerce scale"""

        optimization_plan = {
            "current_metrics": {
                "page_load_time": 3.2,  # seconds
                "api_response_time": 250,  # ms
                "concurrent_users": 1000,
            },
            "optimizations": {
                "frontend": [
                    "Implement lazy loading for product images",
                    "Use WebP format with fallback",
                    "Enable HTTP/2 push for critical resources",
                    "Implement service worker for offline functionality",
                ],
                "backend": [
                    "Implement database connection pooling",
                    "Add Redis caching for product catalog",
                    "Use Elasticsearch for search functionality",
                    "Implement GraphQL for efficient data fetching",
                ],
                "infrastructure": [
                    "Configure CDN for static assets",
                    "Set up auto-scaling groups",
                    "Implement blue-green deployment",
                    "Use container orchestration (Kubernetes)",
                ],
            },
            "expected_improvements": {
                "page_load_time": 1.2,  # seconds (62% improvement)
                "api_response_time": 50,  # ms (80% improvement)
                "concurrent_users": 10000,  # 10x increase
            },
            "caching_strategy": {
                "browser_cache": "1 year for static assets",
                "cdn_cache": "24 hours for product images",
                "application_cache": {
                    "product_catalog": "1 hour TTL",
                    "user_sessions": "30 minutes TTL",
                    "search_results": "15 minutes TTL",
                },
            },
        }

        return {
            "task_id": task.id,
            "optimization_plan": optimization_plan,
            "implementation_priority": [
                "CDN configuration (immediate impact)",
                "Database optimization (high impact)",
                "Caching implementation (medium-term)",
                "Frontend optimization (progressive)",
            ],
            "estimated_performance_gain": "70% overall improvement",
        }

    async def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from performance metrics and bottlenecks"""
        self.memory.append(experience)

        if "actual_improvements" in experience:
            # Compare predicted vs actual improvements
            for metric, actual in experience["actual_improvements"].items():
                predicted = experience.get("predicted_improvements", {}).get(metric, 0)
                accuracy = 1 - abs(actual - predicted) / max(actual, predicted)
                self.performance_metrics[f"prediction_accuracy_{metric}"].append(
                    accuracy
                )


async def run_ecommerce_project():
    """Execute a complete e-commerce platform project"""

    # Initialize the engine
    engine = AutonomousProjectEngine(storage_path="gs://ecommerce-project-storage")

    # Register specialized agents
    db_agent = DatabaseArchitectAgent("db_architect_01")
    security_agent = SecurityAuditAgent("security_auditor_01")
    perf_agent = PerformanceOptimizationAgent("performance_optimizer_01")

    engine.orchestrator.register_agent(db_agent)
    engine.orchestrator.register_agent(security_agent)
    engine.orchestrator.register_agent(perf_agent)

    # Define the project
    project_result = await engine.execute_project(
        project_description="Build a scalable B2C e-commerce platform with marketplace capabilities",
        objectives=[
            "Support 1M+ products and 100K+ daily active users",
            "Sub-2-second page load times globally",
            "99.9% uptime SLA",
            "PCI-DSS compliant payment processing",
            "Multi-vendor marketplace functionality",
            "Real-time inventory management",
            "AI-powered product recommendations",
        ],
        constraints={
            "budget": 2000000,  # $2M
            "timeline": "6 months",
            "technology_stack": {
                "frontend": ["React", "Next.js", "TypeScript"],
                "backend": ["Python", "FastAPI", "GraphQL"],
                "database": ["PostgreSQL", "Redis", "Elasticsearch"],
                "infrastructure": ["AWS", "Kubernetes", "Terraform"],
            },
            "team_size": 15,
            "regulations": ["GDPR", "CCPA", "PCI-DSS"],
        },
        success_criteria={
            "performance": {
                "page_load_time": "< 2 seconds",
                "api_response_time": "< 100ms p95",
                "concurrent_users": "> 10,000",
            },
            "reliability": {
                "uptime": "> 99.9%",
                "error_rate": "< 0.1%",
                "data_consistency": "100%",
            },
            "security": {
                "vulnerability_score": "< 3.0 CVSS",
                "penetration_test": "Pass",
                "compliance_audit": "Pass",
            },
            "business": {
                "conversion_rate": "> 3%",
                "cart_abandonment": "< 65%",
                "customer_satisfaction": "> 4.5/5",
            },
        },
        quality_standards={
            "code_coverage": 0.85,
            "performance": 0.9,
            "security": 0.95,
            "accessibility": 0.9,
            "documentation": 0.8,
        },
        domain="ecommerce",
        priority=9,
        deadline=datetime.now() + timedelta(days=180),
    )

    # Display comprehensive results
    print("\n" + "=" * 60)
    print("E-COMMERCE PROJECT EXECUTION COMPLETE")
    print("=" * 60)

    print(f"\nProject ID: {project_result['project_id']}")
    print(f"Status: {project_result['status']}")
    print(f"Execution Time: {project_result['execution_time']}")
    print(f"Iterations Required: {project_result['iterations']}")

    print("\nQuality Metrics:")
    for metric, score in project_result["quality_metrics"].items():
        print(f"  - {metric}: {score:.2%}")

    print("\nAgent Performance:")
    for agent_id, performance in project_result["agent_performances"].items():
        print(f"  - {agent_id}: {performance:.2%}")

    print("\nKey Deliverables:")
    if "deliverables" in project_result:
        for deliverable_type, content in project_result["deliverables"].items():
            print(f"  - {deliverable_type}: Generated successfully")

    print("\nLearning Insights:")
    insights = project_result.get("learning_insights", {})
    if insights.get("agent_performances"):
        for agent, perf in insights["agent_performances"].items():
            print(
                f"  - {agent}: {perf.get('trend', 'stable')} "
                f"(current: {perf.get('current', 0):.2%})"
            )

    print("\nRecommendations for Future Projects:")
    for i, rec in enumerate(project_result.get("recommendations", [])[:5], 1):
        print(
            f"  {i}. {rec.get('recommendation', 'N/A')} "
            f"[Priority: {rec.get('priority', 'medium')}]"
        )

    print("\n" + "=" * 60)
    print("Project artifacts saved to cloud storage")
    print("Access dashboard at: https://project-dashboard.example.com")
    print("=" * 60)

    return project_result


if __name__ == "__main__":
    asyncio.run(run_ecommerce_project())
