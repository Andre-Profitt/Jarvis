#!/usr/bin/env python3
"""
Smart cleanup script for JARVIS project
"""

import os
import shutil
from pathlib import Path

def cleanup():
    """Move files to archive"""
    archive_dir = Path(".archive")
    archive_dir.mkdir(exist_ok=True)

    # Archive old launchers
    launcher_archive = archive_dir / "old_launchers"
    launcher_archive.mkdir(exist_ok=True)
    if Path("launch_voice_jarvis.py").exists():
        shutil.move("launch_voice_jarvis.py", launcher_archive / "launch_voice_jarvis.py")
    if Path("start_jarvis_elevenlabs.sh").exists():
        shutil.move("start_jarvis_elevenlabs.sh", launcher_archive / "start_jarvis_elevenlabs.sh")
    if Path("phase9/launch_phase9.py").exists():
        shutil.move("phase9/launch_phase9.py", launcher_archive / "launch_phase9.py")
    if Path(".jarvis_launched").exists():
        shutil.move(".jarvis_launched", launcher_archive / ".jarvis_launched")
    if Path("start_jarvis.py").exists():
        shutil.move("start_jarvis.py", launcher_archive / "start_jarvis.py")
    if Path("launch_enhanced.sh").exists():
        shutil.move("launch_enhanced.sh", launcher_archive / "launch_enhanced.sh")
    if Path("phase2/launch_phase2_demo.py").exists():
        shutil.move("phase2/launch_phase2_demo.py", launcher_archive / "launch_phase2_demo.py")
    if Path("backup/launchers/LAUNCH-JARVIS-UNIFIED.py").exists():
        shutil.move("backup/launchers/LAUNCH-JARVIS-UNIFIED.py", launcher_archive / "LAUNCH-JARVIS-UNIFIED.py")
    if Path("backup/launchers/LAUNCH-JARVIS.py").exists():
        shutil.move("backup/launchers/LAUNCH-JARVIS.py", launcher_archive / "LAUNCH-JARVIS.py")
    if Path("launch_ultimate.sh").exists():
        shutil.move("launch_ultimate.sh", launcher_archive / "launch_ultimate.sh")
    if Path("run_jarvis.py").exists():
        shutil.move("run_jarvis.py", launcher_archive / "run_jarvis.py")
    if Path("phase11-integration/launch_phase11.py").exists():
        shutil.move("phase11-integration/launch_phase11.py", launcher_archive / "launch_phase11.py")
    if Path("jarvis_launch.log").exists():
        shutil.move("jarvis_launch.log", launcher_archive / "jarvis_launch.log")
    if Path("phase8/launch_phase8.py").exists():
        shutil.move("phase8/launch_phase8.py", launcher_archive / "launch_phase8.py")
    if Path("launch_jarvis_ultimate.sh").exists():
        shutil.move("launch_jarvis_ultimate.sh", launcher_archive / "launch_jarvis_ultimate.sh")
    if Path("backup/launchers/LAUNCH-JARVIS-FIXED.py").exists():
        shutil.move("backup/launchers/LAUNCH-JARVIS-FIXED.py", launcher_archive / "LAUNCH-JARVIS-FIXED.py")
    if Path("LAUNCH-JARVIS-REAL.py").exists():
        shutil.move("LAUNCH-JARVIS-REAL.py", launcher_archive / "LAUNCH-JARVIS-REAL.py")
    if Path("launch_jarvis_conversation.py").exists():
        shutil.move("launch_jarvis_conversation.py", launcher_archive / "launch_jarvis_conversation.py")
    if Path("launch_minimal.sh").exists():
        shutil.move("launch_minimal.sh", launcher_archive / "launch_minimal.sh")
    if Path("backup/launchers/LAUNCH-JARVIS-ENHANCED.py").exists():
        shutil.move("backup/launchers/LAUNCH-JARVIS-ENHANCED.py", launcher_archive / "LAUNCH-JARVIS-ENHANCED.py")
    if Path("backup/launchers/launch_jarvis_advanced.py").exists():
        shutil.move("backup/launchers/launch_jarvis_advanced.py", launcher_archive / "launch_jarvis_advanced.py")
    if Path("backup/launchers/LAUNCH-JARVIS-PATCHED.py").exists():
        shutil.move("backup/launchers/LAUNCH-JARVIS-PATCHED.py", launcher_archive / "LAUNCH-JARVIS-PATCHED.py")
    if Path("logs/jarvis_launch_20250629_131241.log").exists():
        shutil.move("logs/jarvis_launch_20250629_131241.log", launcher_archive / "jarvis_launch_20250629_131241.log")
    if Path("start_jarvis.sh").exists():
        shutil.move("start_jarvis.sh", launcher_archive / "start_jarvis.sh")
    if Path("backup/launchers/LAUNCH-JARVIS-FULL.py").exists():
        shutil.move("backup/launchers/LAUNCH-JARVIS-FULL.py", launcher_archive / "LAUNCH-JARVIS-FULL.py")
    if Path("backup/launchers/__init__.py").exists():
        shutil.move("backup/launchers/__init__.py", launcher_archive / "__init__.py")
    if Path("quick_launch.py").exists():
        shutil.move("quick_launch.py", launcher_archive / "quick_launch.py")
    if Path("docs/launcher_migration_guide.md").exists():
        shutil.move("docs/launcher_migration_guide.md", launcher_archive / "launcher_migration_guide.md")
    if Path("launch_wrapper.sh").exists():
        shutil.move("launch_wrapper.sh", launcher_archive / "launch_wrapper.sh")

    # Archive phase files
    phase_archive = archive_dir / "phase_files"
    phase_archive.mkdir(exist_ok=True)
    if Path("setup_phase5.sh").exists():
        shutil.move("setup_phase5.sh", phase_archive / "setup_phase5.sh")
    if Path("PHASE3_COMPLETE.md").exists():
        shutil.move("PHASE3_COMPLETE.md", phase_archive / "PHASE3_COMPLETE.md")
    if Path("PHASE10_COMPLETE.md").exists():
        shutil.move("PHASE10_COMPLETE.md", phase_archive / "PHASE10_COMPLETE.md")
    if Path("run_phase12.sh").exists():
        shutil.move("run_phase12.sh", phase_archive / "run_phase12.sh")
    if Path("htmlcov/z_57760688d1f824db_phase5_integration_py.html").exists():
        shutil.move("htmlcov/z_57760688d1f824db_phase5_integration_py.html", phase_archive / "z_57760688d1f824db_phase5_integration_py.html")
    if Path("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_jarvis_phase6_integration_py.html").exists():
        shutil.move("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_jarvis_phase6_integration_py.html", phase_archive / "z_77821a45e9acb8e4_jarvis_phase6_integration_py.html")
    if Path("jarvis-phase4-predictive-dashboard.html").exists():
        shutil.move("jarvis-phase4-predictive-dashboard.html", phase_archive / "jarvis-phase4-predictive-dashboard.html")
    if Path("core/phase5_graduated_interventions.py").exists():
        shutil.move("core/phase5_graduated_interventions.py", phase_archive / "phase5_graduated_interventions.py")
    if Path("run_phase3.sh").exists():
        shutil.move("run_phase3.sh", phase_archive / "run_phase3.sh")
    if Path("run_phase10.sh").exists():
        shutil.move("run_phase10.sh", phase_archive / "run_phase10.sh")
    if Path("htmlcov/z_57760688d1f824db_jarvis_phase7_integration_py.html").exists():
        shutil.move("htmlcov/z_57760688d1f824db_jarvis_phase7_integration_py.html", phase_archive / "z_57760688d1f824db_jarvis_phase7_integration_py.html")
    if Path("htmlcov/z_57760688d1f824db_jarvis_phase6_integration_py.html").exists():
        shutil.move("htmlcov/z_57760688d1f824db_jarvis_phase6_integration_py.html", phase_archive / "z_57760688d1f824db_jarvis_phase6_integration_py.html")
    if Path("run_phase1.sh").exists():
        shutil.move("run_phase1.sh", phase_archive / "run_phase1.sh")
    if Path("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_phase5_emotional_continuity_py.html").exists():
        shutil.move("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_phase5_emotional_continuity_py.html", phase_archive / "z_77821a45e9acb8e4_phase5_emotional_continuity_py.html")
    if Path("htmlcov/z_57760688d1f824db_phase5_graduated_interventions_py.html").exists():
        shutil.move("htmlcov/z_57760688d1f824db_phase5_graduated_interventions_py.html", phase_archive / "z_57760688d1f824db_phase5_graduated_interventions_py.html")
    if Path("PHASE5_CUSTOMIZATION_GUIDE.md").exists():
        shutil.move("PHASE5_CUSTOMIZATION_GUIDE.md", phase_archive / "PHASE5_CUSTOMIZATION_GUIDE.md")
    if Path("PHASE1_COMPLETE.md").exists():
        shutil.move("PHASE1_COMPLETE.md", phase_archive / "PHASE1_COMPLETE.md")
    if Path("run_phase4.sh").exists():
        shutil.move("run_phase4.sh", phase_archive / "run_phase4.sh")
    if Path("PHASE1_PERFORMANCE_GUIDE.md").exists():
        shutil.move("PHASE1_PERFORMANCE_GUIDE.md", phase_archive / "PHASE1_PERFORMANCE_GUIDE.md")
    if Path("PHASE2_IMPLEMENTATION_SUMMARY.md").exists():
        shutil.move("PHASE2_IMPLEMENTATION_SUMMARY.md", phase_archive / "PHASE2_IMPLEMENTATION_SUMMARY.md")
    if Path("PHASE4_COMPLETE.md").exists():
        shutil.move("PHASE4_COMPLETE.md", phase_archive / "PHASE4_COMPLETE.md")
    if Path("core/phase5_integration.py").exists():
        shutil.move("core/phase5_integration.py", phase_archive / "phase5_integration.py")
    if Path("PHASE6_README.md").exists():
        shutil.move("PHASE6_README.md", phase_archive / "PHASE6_README.md")
    if Path("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_jarvis_phase7_integration_py.html").exists():
        shutil.move("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_jarvis_phase7_integration_py.html", phase_archive / "z_77821a45e9acb8e4_jarvis_phase7_integration_py.html")
    if Path("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_phase5_integration_py.html").exists():
        shutil.move("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_phase5_integration_py.html", phase_archive / "z_77821a45e9acb8e4_phase5_integration_py.html")
    if Path("run_phase2.sh").exists():
        shutil.move("run_phase2.sh", phase_archive / "run_phase2.sh")
    if Path("core/jarvis_phase7_integration.py").exists():
        shutil.move("core/jarvis_phase7_integration.py", phase_archive / "jarvis_phase7_integration.py")
    if Path("docs/PHASE3_INTEGRATION_GUIDE.md").exists():
        shutil.move("docs/PHASE3_INTEGRATION_GUIDE.md", phase_archive / "PHASE3_INTEGRATION_GUIDE.md")
    if Path("core/__pycache__/phase5_graduated_interventions.cpython-39.pyc").exists():
        shutil.move("core/__pycache__/phase5_graduated_interventions.cpython-39.pyc", phase_archive / "phase5_graduated_interventions.cpython-39.pyc")
    if Path("PHASE7_README.md").exists():
        shutil.move("PHASE7_README.md", phase_archive / "PHASE7_README.md")
    if Path("core/phase5_emotional_continuity.py").exists():
        shutil.move("core/phase5_emotional_continuity.py", phase_archive / "phase5_emotional_continuity.py")
    if Path("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_phase5_graduated_interventions_py.html").exists():
        shutil.move("mcp_servers/claude-memory-rag/htmlcov/z_77821a45e9acb8e4_phase5_graduated_interventions_py.html", phase_archive / "z_77821a45e9acb8e4_phase5_graduated_interventions_py.html")
    if Path("run_phase6.sh").exists():
        shutil.move("run_phase6.sh", phase_archive / "run_phase6.sh")
    if Path("core/__pycache__/phase5_integration.cpython-39.pyc").exists():
        shutil.move("core/__pycache__/phase5_integration.cpython-39.pyc", phase_archive / "phase5_integration.cpython-39.pyc")
    if Path("core/jarvis_phase6_integration.py").exists():
        shutil.move("core/jarvis_phase6_integration.py", phase_archive / "jarvis_phase6_integration.py")
    if Path("core/__pycache__/jarvis_phase6_integration.cpython-39.pyc").exists():
        shutil.move("core/__pycache__/jarvis_phase6_integration.cpython-39.pyc", phase_archive / "jarvis_phase6_integration.cpython-39.pyc")
    if Path("core/__pycache__/jarvis_phase7_integration.cpython-39.pyc").exists():
        shutil.move("core/__pycache__/jarvis_phase7_integration.cpython-39.pyc", phase_archive / "jarvis_phase7_integration.cpython-39.pyc")
    if Path("core/__pycache__/phase5_emotional_continuity.cpython-39.pyc").exists():
        shutil.move("core/__pycache__/phase5_emotional_continuity.cpython-39.pyc", phase_archive / "phase5_emotional_continuity.cpython-39.pyc")
    if Path("jarvis-phase1-monitor.html").exists():
        shutil.move("jarvis-phase1-monitor.html", phase_archive / "jarvis-phase1-monitor.html")
    if Path("docs/PHASE12_INTEGRATION_GUIDE.md").exists():
        shutil.move("docs/PHASE12_INTEGRATION_GUIDE.md", phase_archive / "PHASE12_INTEGRATION_GUIDE.md")
    if Path("PHASE5_FINAL_SUMMARY.md").exists():
        shutil.move("PHASE5_FINAL_SUMMARY.md", phase_archive / "PHASE5_FINAL_SUMMARY.md")
    if Path("docs/PHASE1_INTEGRATION_GUIDE.md").exists():
        shutil.move("docs/PHASE1_INTEGRATION_GUIDE.md", phase_archive / "PHASE1_INTEGRATION_GUIDE.md")
    if Path("PHASE5_COMPLETE.md").exists():
        shutil.move("PHASE5_COMPLETE.md", phase_archive / "PHASE5_COMPLETE.md")
    if Path("htmlcov/z_57760688d1f824db_phase5_emotional_continuity_py.html").exists():
        shutil.move("htmlcov/z_57760688d1f824db_phase5_emotional_continuity_py.html", phase_archive / "z_57760688d1f824db_phase5_emotional_continuity_py.html")

    # Delete temp files
    if Path("core/__pycache__/metacognitive_introspector.cpython-39.pyc").exists():
        Path("core/__pycache__/metacognitive_introspector.cpython-39.pyc").unlink()
    if Path("jarvis.db").exists():
        Path("jarvis.db").unlink()
    if Path("core/__pycache__/world_class_ml.cpython-39.pyc").exists():
        Path("core/__pycache__/world_class_ml.cpython-39.pyc").unlink()
    if Path("core/__pycache__/model_nursery.cpython-39.pyc").exists():
        Path("core/__pycache__/model_nursery.cpython-39.pyc").unlink()
    if Path("core/__pycache__/performance_optimizer.cpython-39.pyc").exists():
        Path("core/__pycache__/performance_optimizer.cpython-39.pyc").unlink()
    if Path("tools/__pycache__/__init__.cpython-39.pyc").exists():
        Path("tools/__pycache__/__init__.cpython-39.pyc").unlink()
    if Path("core/__pycache__/natural_interaction_core.cpython-39.pyc").exists():
        Path("core/__pycache__/natural_interaction_core.cpython-39.pyc").unlink()
    if Path("core/base/__pycache__/__init__.cpython-39.pyc").exists():
        Path("core/base/__pycache__/__init__.cpython-39.pyc").unlink()
    if Path("core/__pycache__/real_ml_training.cpython-39.pyc").exists():
        Path("core/__pycache__/real_ml_training.cpython-39.pyc").unlink()
    if Path("core/__pycache__/__init__.cpython-39.pyc").exists():
        Path("core/__pycache__/__init__.cpython-39.pyc").unlink()
    if Path("core/__pycache__/contract_net_protocol.cpython-39.pyc").exists():
        Path("core/__pycache__/contract_net_protocol.cpython-39.pyc").unlink()
    if Path("__pycache__/jarvis_live.cpython-313.pyc").exists():
        Path("__pycache__/jarvis_live.cpython-313.pyc").unlink()
    if Path("core/__pycache__/ultimate_project_autonomy.cpython-39.pyc").exists():
        Path("core/__pycache__/ultimate_project_autonomy.cpython-39.pyc").unlink()
    if Path("__pycache__/jarvis_consciousness.cpython-313.pyc").exists():
        Path("__pycache__/jarvis_consciousness.cpython-313.pyc").unlink()
    if Path("__pycache__/long_term_memory.cpython-313.pyc").exists():
        Path("__pycache__/long_term_memory.cpython-313.pyc").unlink()
    if Path("coverage.xml").exists():
        Path("coverage.xml").unlink()
    if Path("core/__pycache__/conversational_memory.cpython-39.pyc").exists():
        Path("core/__pycache__/conversational_memory.cpython-39.pyc").unlink()
    if Path("core/__pycache__/voice_recognition.cpython-39.pyc").exists():
        Path("core/__pycache__/voice_recognition.cpython-39.pyc").unlink()
    if Path("core/__pycache__/agent_registry.cpython-39.pyc").exists():
        Path("core/__pycache__/agent_registry.cpython-39.pyc").unlink()
    if Path("core/base/__pycache__/component.cpython-39.pyc").exists():
        Path("core/base/__pycache__/component.cpython-39.pyc").unlink()
    if Path("logs/jarvis_now.log").exists():
        Path("logs/jarvis_now.log").unlink()
    if Path("core/__pycache__/autonomous_project_engine.cpython-313.pyc").exists():
        Path("core/__pycache__/autonomous_project_engine.cpython-313.pyc").unlink()
    if Path("core/__pycache__/monitoring.cpython-39.pyc").exists():
        Path("core/__pycache__/monitoring.cpython-39.pyc").unlink()
    if Path("tools/__pycache__/communicator.cpython-39.pyc").exists():
        Path("tools/__pycache__/communicator.cpython-39.pyc").unlink()
    if Path("core/__pycache__/security_sandbox.cpython-39.pyc").exists():
        Path("core/__pycache__/security_sandbox.cpython-39.pyc").unlink()
    if Path("core/__pycache__/database.cpython-39.pyc").exists():
        Path("core/__pycache__/database.cpython-39.pyc").unlink()
    if Path("core/__pycache__/quantum_swarm_jarvis.cpython-39.pyc").exists():
        Path("core/__pycache__/quantum_swarm_jarvis.cpython-39.pyc").unlink()
    if Path("core/__pycache__/enhanced_episodic_memory.cpython-313.pyc").exists():
        Path("core/__pycache__/enhanced_episodic_memory.cpython-313.pyc").unlink()
    if Path("core/__pycache__/llm_research_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/llm_research_integration.cpython-39.pyc").unlink()
    if Path("core/__pycache__/working_multi_ai.cpython-39.pyc").exists():
        Path("core/__pycache__/working_multi_ai.cpython-39.pyc").unlink()
    if Path("core/__pycache__/fluid_state_management.cpython-313.pyc").exists():
        Path("core/__pycache__/fluid_state_management.cpython-313.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/project_memory_silos.cpython-39.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/project_memory_silos.cpython-39.pyc").unlink()
    if Path("core/__pycache__/world_class_swarm.cpython-39.pyc").exists():
        Path("core/__pycache__/world_class_swarm.cpython-39.pyc").unlink()
    if Path("core/__pycache__/websocket_security.cpython-39.pyc").exists():
        Path("core/__pycache__/websocket_security.cpython-39.pyc").unlink()
    if Path("core/__pycache__/fusion_improvements.cpython-39.pyc").exists():
        Path("core/__pycache__/fusion_improvements.cpython-39.pyc").unlink()
    if Path("core/__pycache__/memory_enhanced_processing.cpython-39.pyc").exists():
        Path("core/__pycache__/memory_enhanced_processing.cpython-39.pyc").unlink()
    if Path("core/__pycache__/unified_input_pipeline.cpython-39.pyc").exists():
        Path("core/__pycache__/unified_input_pipeline.cpython-39.pyc").unlink()
    if Path("core/__pycache__/neural_resource_manager.cpython-313.pyc").exists():
        Path("core/__pycache__/neural_resource_manager.cpython-313.pyc").unlink()
    if Path("core/__pycache__/redis_cache_layer.cpython-39.pyc").exists():
        Path("core/__pycache__/redis_cache_layer.cpython-39.pyc").unlink()
    if Path("tools/__pycache__/scheduler.cpython-39.pyc").exists():
        Path("tools/__pycache__/scheduler.cpython-39.pyc").unlink()
    if Path("core/__pycache__/mcp_integrator.cpython-39.pyc").exists():
        Path("core/__pycache__/mcp_integrator.cpython-39.pyc").unlink()
    if Path("tools/__pycache__/base.cpython-39.pyc").exists():
        Path("tools/__pycache__/base.cpython-39.pyc").unlink()
    if Path("core/__pycache__/neural_resource_simple.cpython-313.pyc").exists():
        Path("core/__pycache__/neural_resource_simple.cpython-313.pyc").unlink()
    if Path("__pycache__/jarvis_seamless_v2.cpython-313.pyc").exists():
        Path("__pycache__/jarvis_seamless_v2.cpython-313.pyc").unlink()
    if Path("core/__pycache__/intelligent_cache.cpython-39.pyc").exists():
        Path("core/__pycache__/intelligent_cache.cpython-39.pyc").unlink()
    if Path("core/__pycache__/llm_research_jarvis.cpython-39.pyc").exists():
        Path("core/__pycache__/llm_research_jarvis.cpython-39.pyc").unlink()
    if Path("jarvis_debug.log").exists():
        Path("jarvis_debug.log").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server_simple_working.cpython-313.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server_simple_working.cpython-313.pyc").unlink()
    if Path("core/__pycache__/configuration.cpython-39.pyc").exists():
        Path("core/__pycache__/configuration.cpython-39.pyc").unlink()
    if Path("logs/jarvis_enhanced_20250629_134459.log").exists():
        Path("logs/jarvis_enhanced_20250629_134459.log").unlink()
    if Path("core/__pycache__/cloud_scale_autonomy_enhanced.cpython-39.pyc").exists():
        Path("core/__pycache__/cloud_scale_autonomy_enhanced.cpython-39.pyc").unlink()
    if Path("core/__pycache__/real_elevenlabs_integration.cpython-313.pyc").exists():
        Path("core/__pycache__/real_elevenlabs_integration.cpython-313.pyc").unlink()
    if Path("jarvis_live.log").exists():
        Path("jarvis_live.log").unlink()
    if Path("core/__pycache__/consciousness_simulation.cpython-313.pyc").exists():
        Path("core/__pycache__/consciousness_simulation.cpython-313.pyc").unlink()
    if Path("__pycache__/__init__.cpython-313.pyc").exists():
        Path("__pycache__/__init__.cpython-313.pyc").unlink()
    if Path("core/__pycache__/natural_language_flow.cpython-39.pyc").exists():
        Path("core/__pycache__/natural_language_flow.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/__init__.py").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/__init__.py").unlink()
    if Path("mcp_servers/claude-memory-rag/htmlcov/coverage_html_cb_6fb7b396.js").exists():
        Path("mcp_servers/claude-memory-rag/htmlcov/coverage_html_cb_6fb7b396.js").unlink()
    if Path("core/__pycache__/parallel_processor.cpython-39.pyc").exists():
        Path("core/__pycache__/parallel_processor.cpython-39.pyc").unlink()
    if Path("core/__pycache__/consciousness_jarvis.cpython-313.pyc").exists():
        Path("core/__pycache__/consciousness_jarvis.cpython-313.pyc").unlink()
    if Path("core/__pycache__/metacognitive_introspector.cpython-313.pyc").exists():
        Path("core/__pycache__/metacognitive_introspector.cpython-313.pyc").unlink()
    if Path("core/__pycache__/elite_proactive_assistant_backup.cpython-39.pyc").exists():
        Path("core/__pycache__/elite_proactive_assistant_backup.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server_claude_powered.cpython-39.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server_claude_powered.cpython-39.pyc").unlink()
    if Path("core/__pycache__/lazy_loader.cpython-39.pyc").exists():
        Path("core/__pycache__/lazy_loader.cpython-39.pyc").unlink()
    if Path(".coveragerc").exists():
        Path(".coveragerc").unlink()
    if Path("core/__pycache__/self_healing_system.cpython-313.pyc").exists():
        Path("core/__pycache__/self_healing_system.cpython-313.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server_hybrid_storage.cpython-39.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server_hybrid_storage.cpython-39.pyc").unlink()
    if Path("core/__pycache__/resource_manager.cpython-39.pyc").exists():
        Path("core/__pycache__/resource_manager.cpython-39.pyc").unlink()
    if Path("core/__pycache__/neural_resource_simple.cpython-39.pyc").exists():
        Path("core/__pycache__/neural_resource_simple.cpython-39.pyc").unlink()
    if Path("core/__pycache__/fusion_scenarios.cpython-39.pyc").exists():
        Path("core/__pycache__/fusion_scenarios.cpython-39.pyc").unlink()
    if Path("core/__pycache__/real_claude_integration.cpython-313.pyc").exists():
        Path("core/__pycache__/real_claude_integration.cpython-313.pyc").unlink()
    if Path("core/__pycache__/unified_input_pipeline.cpython-313.pyc").exists():
        Path("core/__pycache__/unified_input_pipeline.cpython-313.pyc").unlink()
    if Path("core/base/__pycache__/integration.cpython-39.pyc").exists():
        Path("core/base/__pycache__/integration.cpython-39.pyc").unlink()
    if Path("core/__pycache__/health_checks.cpython-313.pyc").exists():
        Path("core/__pycache__/health_checks.cpython-313.pyc").unlink()
    if Path("core/__pycache__/monitoring.cpython-313.pyc").exists():
        Path("core/__pycache__/monitoring.cpython-313.pyc").unlink()
    if Path("core/__pycache__/emotional_continuity.cpython-39.pyc").exists():
        Path("core/__pycache__/emotional_continuity.cpython-39.pyc").unlink()
    if Path("core/__pycache__/real_claude_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/real_claude_integration.cpython-39.pyc").unlink()
    if Path("core/__pycache__/unified_claude_protocol.cpython-39.pyc").exists():
        Path("core/__pycache__/unified_claude_protocol.cpython-39.pyc").unlink()
    if Path("core/__pycache__/jarvis_enhanced_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/jarvis_enhanced_integration.cpython-39.pyc").unlink()
    if Path("core/consciousness/__pycache__/__init__.cpython-39.pyc").exists():
        Path("core/consciousness/__pycache__/__init__.cpython-39.pyc").unlink()
    if Path("core/__pycache__/predictive_preloading_system.cpython-39.pyc").exists():
        Path("core/__pycache__/predictive_preloading_system.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server_simple_working.cpython-39.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server_simple_working.cpython-39.pyc").unlink()
    if Path("core/__pycache__/jarvis_memory.cpython-39.pyc").exists():
        Path("core/__pycache__/jarvis_memory.cpython-39.pyc").unlink()
    if Path("__pycache__/__init__.cpython-39.pyc").exists():
        Path("__pycache__/__init__.cpython-39.pyc").unlink()
    if Path("core/__pycache__/jarvis_enhanced_core.cpython-39.pyc").exists():
        Path("core/__pycache__/jarvis_enhanced_core.cpython-39.pyc").unlink()
    if Path("core/__pycache__/jarvis_enhanced_core.cpython-313.pyc").exists():
        Path("core/__pycache__/jarvis_enhanced_core.cpython-313.pyc").unlink()
    if Path("core/__pycache__/enhanced_privacy_learning.cpython-39.pyc").exists():
        Path("core/__pycache__/enhanced_privacy_learning.cpython-39.pyc").unlink()
    if Path("core/__pycache__/self_healing_system.cpython-39.pyc").exists():
        Path("core/__pycache__/self_healing_system.cpython-39.pyc").unlink()
    if Path("core/__pycache__/jarvis_specialized_agents.cpython-39.pyc").exists():
        Path("core/__pycache__/jarvis_specialized_agents.cpython-39.pyc").unlink()
    if Path("core/__pycache__/self_healing_dashboard.cpython-39.pyc").exists():
        Path("core/__pycache__/self_healing_dashboard.cpython-39.pyc").unlink()
    if Path("core/__pycache__/gcs_storage.cpython-313.pyc").exists():
        Path("core/__pycache__/gcs_storage.cpython-313.pyc").unlink()
    if Path("core/__pycache__/tools_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/tools_integration.cpython-39.pyc").unlink()
    if Path("__pycache__/missing_components.cpython-313.pyc").exists():
        Path("__pycache__/missing_components.cpython-313.pyc").unlink()
    if Path("core/__pycache__/elite_proactive_assistant.cpython-39.pyc").exists():
        Path("core/__pycache__/elite_proactive_assistant.cpython-39.pyc").unlink()
    if Path("core/__pycache__/architecture_evolver.cpython-39.pyc").exists():
        Path("core/__pycache__/architecture_evolver.cpython-39.pyc").unlink()
    if Path("core/__pycache__/minimal_jarvis.cpython-39.pyc").exists():
        Path("core/__pycache__/minimal_jarvis.cpython-39.pyc").unlink()
    if Path("core/__pycache__/self_healing_simple.cpython-39.pyc").exists():
        Path("core/__pycache__/self_healing_simple.cpython-39.pyc").unlink()
    if Path("core/__pycache__/context_persistence_manager.cpython-39.pyc").exists():
        Path("core/__pycache__/context_persistence_manager.cpython-39.pyc").unlink()
    if Path("htmlcov/coverage_html_cb_6fb7b396.js").exists():
        Path("htmlcov/coverage_html_cb_6fb7b396.js").unlink()
    if Path("jarvis_memory.db").exists():
        Path("jarvis_memory.db").unlink()
    if Path("tools/__pycache__/knowledge_base.cpython-39.pyc").exists():
        Path("tools/__pycache__/knowledge_base.cpython-39.pyc").unlink()
    if Path("core/__pycache__/neural_resource_manager.cpython-39.pyc").exists():
        Path("core/__pycache__/neural_resource_manager.cpython-39.pyc").unlink()
    if Path("core/__pycache__/consciousness_extensions.cpython-39.pyc").exists():
        Path("core/__pycache__/consciousness_extensions.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server_enhanced.cpython-39.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server_enhanced.cpython-39.pyc").unlink()
    if Path("core/__pycache__/gcs_storage.cpython-39.pyc").exists():
        Path("core/__pycache__/gcs_storage.cpython-39.pyc").unlink()
    if Path("core/__pycache__/updated_multi_ai_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/updated_multi_ai_integration.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/.coverage").exists():
        Path("mcp_servers/claude-memory-rag/.coverage").unlink()
    if Path("jarvis_output.log").exists():
        Path("jarvis_output.log").unlink()
    if Path("core/__pycache__/program_synthesis_engine.cpython-39.pyc").exists():
        Path("core/__pycache__/program_synthesis_engine.cpython-39.pyc").unlink()
    if Path("core/__pycache__/self_healing_simple.cpython-313.pyc").exists():
        Path("core/__pycache__/self_healing_simple.cpython-313.pyc").unlink()
    if Path("core/__pycache__/tool_deployment_system.cpython-39.pyc").exists():
        Path("core/__pycache__/tool_deployment_system.cpython-39.pyc").unlink()
    if Path("core/__pycache__/consciousness_simulation.cpython-39.pyc").exists():
        Path("core/__pycache__/consciousness_simulation.cpython-39.pyc").unlink()
    if Path("core/__pycache__/emotional_intelligence.cpython-39.pyc").exists():
        Path("core/__pycache__/emotional_intelligence.cpython-39.pyc").unlink()
    if Path("core/__pycache__/voice_system.cpython-39.pyc").exists():
        Path("core/__pycache__/voice_system.cpython-39.pyc").unlink()
    if Path("logs/jarvis_stable.log").exists():
        Path("logs/jarvis_stable.log").unlink()
    if Path("logs/jarvis_deployment.log").exists():
        Path("logs/jarvis_deployment.log").unlink()
    if Path("core/__pycache__/quantum_swarm_optimization.cpython-313.pyc").exists():
        Path("core/__pycache__/quantum_swarm_optimization.cpython-313.pyc").unlink()
    if Path("core/__pycache__/model_ensemble.cpython-39.pyc").exists():
        Path("core/__pycache__/model_ensemble.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server_full_featured.cpython-39.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server_full_featured.cpython-39.pyc").unlink()
    if Path("core/__pycache__/autonomous_project_engine.cpython-39.pyc").exists():
        Path("core/__pycache__/autonomous_project_engine.cpython-39.pyc").unlink()
    if Path("core/__pycache__/code_generator_agent.cpython-39.pyc").exists():
        Path("core/__pycache__/code_generator_agent.cpython-39.pyc").unlink()
    if Path("core/__pycache__/privacy_preserving_learning.cpython-39.pyc").exists():
        Path("core/__pycache__/privacy_preserving_learning.cpython-39.pyc").unlink()
    if Path("core/__pycache__/health_checks.cpython-39.pyc").exists():
        Path("core/__pycache__/health_checks.cpython-39.pyc").unlink()
    if Path("core/__pycache__/real_openai_integration.cpython-313.pyc").exists():
        Path("core/__pycache__/real_openai_integration.cpython-313.pyc").unlink()
    if Path("core/__pycache__/enhanced_episodic_memory.cpython-39.pyc").exists():
        Path("core/__pycache__/enhanced_episodic_memory.cpython-39.pyc").unlink()
    if Path("core/__pycache__/elite_proactive_assistant_v2.cpython-39.pyc").exists():
        Path("core/__pycache__/elite_proactive_assistant_v2.cpython-39.pyc").unlink()
    if Path("core/__pycache__/world_class_ml.cpython-313.pyc").exists():
        Path("core/__pycache__/world_class_ml.cpython-313.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server.cpython-39.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server.cpython-39.pyc").unlink()
    if Path("core/__pycache__/fluid_state_management.cpython-39.pyc").exists():
        Path("core/__pycache__/fluid_state_management.cpython-39.pyc").unlink()
    if Path("core/__pycache__/quantum_swarm_optimization.cpython-39.pyc").exists():
        Path("core/__pycache__/quantum_swarm_optimization.cpython-39.pyc").unlink()
    if Path("core/__pycache__/multimodal_fusion.cpython-39.pyc").exists():
        Path("core/__pycache__/multimodal_fusion.cpython-39.pyc").unlink()
    if Path("core/__pycache__/code_improver.cpython-39.pyc").exists():
        Path("core/__pycache__/code_improver.cpython-39.pyc").unlink()
    if Path("core/__pycache__/consciousness_v2.cpython-39.pyc").exists():
        Path("core/__pycache__/consciousness_v2.cpython-39.pyc").unlink()
    if Path("core/__pycache__/jarvis_ultra_core.cpython-39.pyc").exists():
        Path("core/__pycache__/jarvis_ultra_core.cpython-39.pyc").unlink()
    if Path("core/__pycache__/self_healing_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/self_healing_integration.cpython-39.pyc").unlink()
    if Path("core/__pycache__/ast_analyzer.cpython-39.pyc").exists():
        Path("core/__pycache__/ast_analyzer.cpython-39.pyc").unlink()
    if Path("core/__pycache__/real_openai_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/real_openai_integration.cpython-39.pyc").unlink()
    if Path("core/__pycache__/quantum_swarm_jarvis.cpython-313.pyc").exists():
        Path("core/__pycache__/quantum_swarm_jarvis.cpython-313.pyc").unlink()
    if Path("logs/jarvis_minimal_20250629_131509.log").exists():
        Path("logs/jarvis_minimal_20250629_131509.log").unlink()
    if Path("core/__pycache__/simple_performance_optimizer.cpython-39.pyc").exists():
        Path("core/__pycache__/simple_performance_optimizer.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/jarvis.db").exists():
        Path("mcp_servers/claude-memory-rag/jarvis.db").unlink()
    if Path("core/__pycache__/performance_profiler.cpython-39.pyc").exists():
        Path("core/__pycache__/performance_profiler.cpython-39.pyc").unlink()
    if Path("core/__pycache__/database.cpython-313.pyc").exists():
        Path("core/__pycache__/database.cpython-313.pyc").unlink()
    if Path("core/__pycache__/autonomous_tool_factory.cpython-39.pyc").exists():
        Path("core/__pycache__/autonomous_tool_factory.cpython-39.pyc").unlink()
    if Path("core/__pycache__/config_manager.cpython-39.pyc").exists():
        Path("core/__pycache__/config_manager.cpython-39.pyc").unlink()
    if Path("core/__pycache__/advanced_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/advanced_integration.cpython-39.pyc").unlink()
    if Path("core/__pycache__/visual_feedback_system.cpython-39.pyc").exists():
        Path("core/__pycache__/visual_feedback_system.cpython-39.pyc").unlink()
    if Path("core/__pycache__/working_multi_ai.cpython-313.pyc").exists():
        Path("core/__pycache__/working_multi_ai.cpython-313.pyc").unlink()
    if Path("jarvis.pid").exists():
        Path("jarvis.pid").unlink()
    if Path("core/__pycache__/predictive_monitoring_server.cpython-39.pyc").exists():
        Path("core/__pycache__/predictive_monitoring_server.cpython-39.pyc").unlink()
    if Path("core/__pycache__/llm_research_quickstart.cpython-39.pyc").exists():
        Path("core/__pycache__/llm_research_quickstart.cpython-39.pyc").unlink()
    if Path("core/__pycache__/voice_system.cpython-313.pyc").exists():
        Path("core/__pycache__/voice_system.cpython-313.pyc").unlink()
    if Path("core/__pycache__/jit_compiler.cpython-39.pyc").exists():
        Path("core/__pycache__/jit_compiler.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/coverage.xml").exists():
        Path("mcp_servers/claude-memory-rag/coverage.xml").unlink()
    if Path("core/__pycache__/knowledge_synthesizer.cpython-39.pyc").exists():
        Path("core/__pycache__/knowledge_synthesizer.cpython-39.pyc").unlink()
    if Path("core/__pycache__/__init__.cpython-313.pyc").exists():
        Path("core/__pycache__/__init__.cpython-313.pyc").unlink()
    if Path("core/__pycache__/updated_multi_ai_integration.cpython-313.pyc").exists():
        Path("core/__pycache__/updated_multi_ai_integration.cpython-313.pyc").unlink()
    if Path("core/__pycache__/advanced_self_optimizer.cpython-39.pyc").exists():
        Path("core/__pycache__/advanced_self_optimizer.cpython-39.pyc").unlink()
    if Path("core/__pycache__/performance_tracker.cpython-39.pyc").exists():
        Path("core/__pycache__/performance_tracker.cpython-39.pyc").unlink()
    if Path("jarvis.log").exists():
        Path("jarvis.log").unlink()
    if Path("core/__pycache__/real_elevenlabs_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/real_elevenlabs_integration.cpython-39.pyc").unlink()
    if Path("core/__pycache__/consciousness_jarvis.cpython-39.pyc").exists():
        Path("core/__pycache__/consciousness_jarvis.cpython-39.pyc").unlink()
    if Path("__pycache__/missing_components.cpython-39.pyc").exists():
        Path("__pycache__/missing_components.cpython-39.pyc").unlink()
    if Path("core/__pycache__/minimal_jarvis.cpython-313.pyc").exists():
        Path("core/__pycache__/minimal_jarvis.cpython-313.pyc").unlink()
    if Path("jarvis_full_output.log").exists():
        Path("jarvis_full_output.log").unlink()
    if Path("core/__pycache__/predictive_jarvis_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/predictive_jarvis_integration.cpython-39.pyc").unlink()
    if Path("mcp_servers/claude-memory-rag/__pycache__/server_robust.cpython-313.pyc").exists():
        Path("mcp_servers/claude-memory-rag/__pycache__/server_robust.cpython-313.pyc").unlink()
    if Path("core/__pycache__/predictive_intelligence.cpython-39.pyc").exists():
        Path("core/__pycache__/predictive_intelligence.cpython-39.pyc").unlink()
    if Path("core/__pycache__/metacognitive_jarvis.cpython-39.pyc").exists():
        Path("core/__pycache__/metacognitive_jarvis.cpython-39.pyc").unlink()
    if Path("core/__pycache__/neural_integration.cpython-39.pyc").exists():
        Path("core/__pycache__/neural_integration.cpython-39.pyc").unlink()
    if Path("check_coverage_progress.py").exists():
        Path("check_coverage_progress.py").unlink()
    if Path("__pycache__/__init__.py").exists():
        Path("__pycache__/__init__.py").unlink()

    print("✅ Cleanup complete!")

if __name__ == "__main__":
    cleanup()
