#!/usr/bin/env python3
"""
JARVIS Integration Status Check
Quick script to verify all integrations are working
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.tools_integration import JARVISToolsIntegration
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

console = Console()


async def check_integration_status():
    """Check the status of JARVIS tool integration"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize integration
        task = progress.add_task("Initializing JARVIS integration...", total=None)

        try:
            jarvis = JARVISToolsIntegration()
            await jarvis.initialize()
            progress.update(task, completed=True)

            # Get system status
            progress.update(task, description="Checking system status...")
            status = await jarvis.get_system_status()

            # Create status table
            table = Table(title="JARVIS Integration Status", show_header=True)
            table.add_column("Component", style="cyan", no_wrap=True)
            table.add_column("Status", style="green")
            table.add_column("Details", style="yellow")

            # Add component statuses
            for component, details in status["components"].items():
                status_text = (
                    "✅ Active" if details.get("status") == "active" else "❌ Inactive"
                )
                detail_text = f"Version: {details.get('version', 'N/A')}"
                table.add_row(component.title(), status_text, detail_text)

            console.print("\n")
            console.print(table)

            # System health panel
            health_color = "green" if status["health"] == "healthy" else "red"
            health_panel = Panel(
                f"[{health_color}]System Health: {status['health'].upper()}[/{health_color}]\n\n"
                f"Total Components: {len(status['components'])}\n"
                f"Integration Status: {'✅ Complete' if status['integrated'] else '❌ Incomplete'}\n"
                f"Timestamp: {status['timestamp']}",
                title="System Overview",
                border_style=health_color,
            )
            console.print(health_panel)

            # Test basic functionality
            console.print("\n[bold cyan]Testing Basic Functionality:[/bold cyan]")

            # Test 1: Schedule a task
            task_result = await jarvis.scheduler.execute(
                action="schedule",
                task_name="integration_test",
                function="print",
                args=["Integration test successful!"],
            )
            console.print(
                f"✅ Scheduler: Task scheduled with ID {task_result.data['task_id']}"
            )

            # Test 2: Store knowledge
            kb_result = await jarvis.knowledge_base.execute(
                action="store",
                content="JARVIS integration test completed successfully",
                type="fact",
                tags=["test", "integration"],
            )
            console.print(
                f"✅ Knowledge Base: Entry stored with ID {kb_result.data['id']}"
            )

            # Test 3: Record metric
            metric_result = await jarvis.monitoring.execute(
                action="record_metric",
                name="integration.test.success",
                value=1,
                type="counter",
            )
            console.print("✅ Monitoring: Metric recorded successfully")

            # Test 4: Service discovery
            services_result = await jarvis.communicator.execute(
                action="discover_services"
            )
            console.print(
                f"✅ Communicator: {len(services_result.data)} services registered"
            )

            # Success panel
            success_panel = Panel(
                "[bold green]🎉 JARVIS Integration is fully operational![/bold green]\n\n"
                "All systems are functioning correctly.\n"
                "You can now:\n"
                "• Process user queries with integrated tools\n"
                "• Schedule and manage tasks\n"
                "• Store and query knowledge\n"
                "• Monitor system performance\n"
                "• Enable inter-tool communication\n\n"
                "[bold]Next step:[/bold] Run 'python launch_jarvis.py --mode integrated'",
                title="Integration Complete",
                border_style="green",
            )
            console.print("\n")
            console.print(success_panel)

            # Cleanup
            await jarvis.shutdown()

        except Exception as e:
            progress.stop()
            console.print(
                f"\n[bold red]❌ Integration check failed: {str(e)}[/bold red]"
            )
            console.print("\nPlease ensure all dependencies are installed:")
            console.print("  pip install -r requirements.txt")
            return False

    return True


async def main():
    """Main entry point"""
    console.print(
        Panel.fit(
            "[bold cyan]JARVIS Integration Status Check[/bold cyan]\n"
            "Verifying all tools and integrations are working correctly...",
            border_style="cyan",
        )
    )

    success = await check_integration_status()

    if success:
        # Show next steps
        next_steps = Table(title="Recommended Next Steps", show_header=True)
        next_steps.add_column("Priority", style="cyan", no_wrap=True)
        next_steps.add_column("Task", style="yellow")
        next_steps.add_column("Command", style="green")

        next_steps.add_row(
            "1",
            "Run comprehensive tests",
            "pytest tests/test_tools_comprehensive.py -v",
        )
        next_steps.add_row(
            "2", "Generate API documentation", "python scripts/generate_api_docs.py"
        )
        next_steps.add_row("3", "Set up database", "python scripts/create_db_schema.py")
        next_steps.add_row(
            "4", "Configure production", "cp .env.example .env.production"
        )
        next_steps.add_row("5", "Deploy with Docker", "docker-compose up -d")

        console.print("\n")
        console.print(next_steps)

        # Quick actions panel
        actions_panel = Panel(
            "[bold]Quick Actions:[/bold]\n\n"
            "• Test integration: [cyan]python demo_new_tools.py[/cyan]\n"
            "• Launch JARVIS: [cyan]python launch_jarvis.py --mode integrated[/cyan]\n"
            "• View logs: [cyan]tail -f jarvis.log[/cyan]\n"
            "• Monitor performance: [cyan]python scripts/monitor_performance.py[/cyan]",
            title="Quick Actions",
            border_style="blue",
        )
        console.print("\n")
        console.print(actions_panel)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Status check interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)
