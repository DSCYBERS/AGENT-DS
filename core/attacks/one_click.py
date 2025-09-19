#!/usr/bin/env python3
"""
Agent DS - One-Click Attack Module
Next-Generation Autonomous Hacker Agent

This module orchestrates comprehensive automated penetration testing
with AI-driven adaptive attack sequencing and hacker terminal aesthetics.

Author: Agent DS Team
Version: 2.0
Date: September 16, 2025
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Rich for enhanced terminal output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich import box

# Import Agent DS modules
from ..auth.authentication import AuthenticationManager
from ..database.db_manager import DatabaseManager
from ..ai.autonomous_engine import AutonomousEngine
from ..security.encryption import EncryptionManager
from .recon import ReconnaissanceEngine
from .web_attack import WebAttackEngine
from .db_exploit import DatabaseExploitEngine
from .admin_login import AdminLoginEngine
from .ai_core import AIAdaptiveCore
from ..reporting.report_generator import ReportGenerator

class OneClickAttackOrchestrator:
    """
    Main orchestrator for One-Click Attack functionality
    Coordinates all attack phases with AI-driven adaptive sequencing
    """
    
    def __init__(self):
        self.console = Console()
        self.start_time = None
        self.target_url = None
        self.attack_results = {}
        self.mission_log = []
        self.safe_mode = False
        self.verbose = False
        
        # Initialize components
        self.auth_manager = AuthenticationManager()
        self.db_manager = DatabaseManager()
        self.autonomous_ai = AutonomousEngine()
        self.encryption = EncryptionManager()
        
        # Attack engines
        self.recon_engine = ReconnaissanceEngine()
        self.web_attack_engine = WebAttackEngine()
        self.db_exploit_engine = DatabaseExploitEngine()
        self.admin_login_engine = AdminLoginEngine()
        self.ai_core = AIAdaptiveCore()
        
        # Reporting
        self.report_generator = ReportGenerator()
        
    def print_ascii_banner(self, phase: str):
        """Print ASCII art banners for different attack phases"""
        banners = {
            "MAIN": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║   █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ██████╗ ███████╗           ║
║  ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ██╔══██╗██╔════╝           ║
║  ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       ██║  ██║███████╗           ║
║  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       ██║  ██║╚════██║           ║
║  ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       ██████╔╝███████║           ║
║  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═════╝ ╚══════╝           ║
║                                                                               ║
║               🔥 ONE-CLICK AUTONOMOUS ATTACK SYSTEM 🔥                        ║
║                     Next-Gen AI Hacker Agent v2.0                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
            """,
            
            "RECON": """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ██████╗ ███████╗ ██████╗ ██████╗ ███╗   ██╗███╗   ██╗ █████╗ ██╗███████╗   ║
║  ██╔══██╗██╔════╝██╔════╝██╔═══██╗████╗  ██║████╗  ██║██╔══██╗██║██╔════╝   ║
║  ██████╔╝█████╗  ██║     ██║   ██║██╔██╗ ██║██╔██╗ ██║███████║██║███████╗   ║
║  ██╔══██╗██╔══╝  ██║     ██║   ██║██║╚██╗██║██║╚██╗██║██╔══██║██║╚════██║   ║
║  ██║  ██║███████╗╚██████╗╚██████╔╝██║ ╚████║██║ ╚████║██║  ██║██║███████║   ║
║  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚══════╝   ║
║                          🔍 TARGET INTELLIGENCE 🔍                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """,
            
            "ATTACK": """
╔══════════════════════════════════════════════════════════════════════════════╗
║   █████╗ ████████╗████████╗ █████╗  ██████╗██╗  ██╗                         ║
║  ██╔══██╗╚══██╔══╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝                         ║
║  ███████║   ██║      ██║   ███████║██║     █████╔╝                          ║
║  ██╔══██║   ██║      ██║   ██╔══██║██║     ██╔═██╗                          ║
║  ██║  ██║   ██║      ██║   ██║  ██║╚██████╗██║  ██╗                         ║
║  ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝                         ║
║                           ⚡ WEB EXPLOITATION ⚡                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """,
            
            "DB_EXPLOIT": """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ██████╗ ██████╗     ███████╗██╗  ██╗██████╗ ██╗      ██████╗ ██╗████████╗  ║
║  ██╔══██╗██╔══██╗    ██╔════╝╚██╗██╔╝██╔══██╗██║     ██╔═══██╗██║╚══██╔══╝  ║
║  ██║  ██║██████╔╝    █████╗   ╚███╔╝ ██████╔╝██║     ██║   ██║██║   ██║     ║
║  ██║  ██║██╔══██╗    ██╔══╝   ██╔██╗ ██╔═══╝ ██║     ██║   ██║██║   ██║     ║
║  ██████╔╝██████╔╝    ███████╗██╔╝ ██╗██║     ███████╗╚██████╔╝██║   ██║     ║
║  ╚═════╝ ╚═════╝     ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝ ╚═╝   ╚═╝     ║
║                            💾 DATABASE BREACH 💾                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """,
            
            "ADMIN_LOGIN": """
╔══════════════════════════════════════════════════════════════════════════════╗
║   █████╗ ██████╗ ███╗   ███╗██╗███╗   ██╗    ██╗      ██████╗  ██████╗ ██╗  ║
║  ██╔══██╗██╔══██╗████╗ ████║██║████╗  ██║    ██║     ██╔═══██╗██╔════╝ ██║  ║
║  ███████║██║  ██║██╔████╔██║██║██╔██╗ ██║    ██║     ██║   ██║██║  ███╗██║  ║
║  ██╔══██║██║  ██║██║╚██╔╝██║██║██║╚██╗██║    ██║     ██║   ██║██║   ██║██║  ║
║  ██║  ██║██████╔╝██║ ╚═╝ ██║██║██║ ╚████║    ███████╗╚██████╔╝╚██████╔╝██║  ║
║  ╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ║
║                            🔐 PRIVILEGE ESCALATION 🔐                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """
        }
        
        banner = banners.get(phase, banners["MAIN"])
        if phase == "MAIN":
            self.console.print(banner, style="bold cyan")
        elif phase == "RECON":
            self.console.print(banner, style="bold yellow")
        elif phase == "ATTACK":
            self.console.print(banner, style="bold red")
        elif phase == "DB_EXPLOIT":
            self.console.print(banner, style="bold magenta")
        elif phase == "ADMIN_LOGIN":
            self.console.print(banner, style="bold green")
    
    def log_mission_event(self, phase: str, event: str, status: str, details: Dict = None):
        """Log mission events with timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "phase": phase,
            "event": event,
            "status": status,
            "details": details or {}
        }
        self.mission_log.append(log_entry)
        
        # Print to console with color coding
        status_colors = {
            "SUCCESS": "green",
            "FAILURE": "red",
            "INFO": "cyan",
            "WARNING": "yellow",
            "CRITICAL": "bold red"
        }
        
        color = status_colors.get(status, "white")
        self.console.print(f"[{timestamp}] [{phase}] {event}", style=color)
        
        if self.verbose and details:
            self.console.print(f"  └─ Details: {details}", style="dim")
    
    async def authenticate_admin(self) -> bool:
        """Perform admin authentication before starting attacks"""
        self.console.print("\n🔐 ADMIN AUTHENTICATION REQUIRED", style="bold yellow")
        self.console.print("═" * 50, style="yellow")
        
        try:
            # Use the existing authentication system
            auth_result = await self.auth_manager.authenticate_interactive()
            
            if auth_result["success"]:
                self.log_mission_event("AUTH", "Admin authentication successful", "SUCCESS")
                self.console.print("✓ Authentication Passed", style="bold green")
                return True
            else:
                self.log_mission_event("AUTH", "Admin authentication failed", "FAILURE")
                self.console.print("✗ Authentication Failed", style="bold red")
                return False
                
        except Exception as e:
            self.log_mission_event("AUTH", f"Authentication error: {str(e)}", "CRITICAL")
            self.console.print(f"✗ Authentication Error: {str(e)}", style="bold red")
            return False
    
    async def run_reconnaissance_phase(self) -> Dict[str, Any]:
        """Execute reconnaissance phase"""
        self.print_ascii_banner("RECON")
        self.log_mission_event("RECON", f"Starting reconnaissance on {self.target_url}", "INFO")
        
        recon_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Subdomain enumeration
            task1 = progress.add_task("🔍 Enumerating subdomains...", total=100)
            try:
                subdomains = await self.recon_engine.enumerate_subdomains(self.target_url)
                recon_results["subdomains"] = subdomains
                progress.update(task1, advance=100)
                self.log_mission_event("RECON", f"Found {len(subdomains)} subdomains", "SUCCESS")
            except Exception as e:
                progress.update(task1, advance=100)
                self.log_mission_event("RECON", f"Subdomain enumeration failed: {str(e)}", "FAILURE")
            
            # Port scanning
            task2 = progress.add_task("🚪 Scanning ports...", total=100)
            try:
                open_ports = await self.recon_engine.scan_ports(self.target_url)
                recon_results["ports"] = open_ports
                progress.update(task2, advance=100)
                self.log_mission_event("RECON", f"Found {len(open_ports)} open ports", "SUCCESS")
            except Exception as e:
                progress.update(task2, advance=100)
                self.log_mission_event("RECON", f"Port scanning failed: {str(e)}", "FAILURE")
            
            # Technology detection
            task3 = progress.add_task("🛠️ Detecting technologies...", total=100)
            try:
                tech_stack = await self.recon_engine.detect_technologies(self.target_url)
                recon_results["technologies"] = tech_stack
                progress.update(task3, advance=100)
                self.log_mission_event("RECON", f"Identified tech stack: {', '.join(tech_stack)}", "SUCCESS")
            except Exception as e:
                progress.update(task3, advance=100)
                self.log_mission_event("RECON", f"Technology detection failed: {str(e)}", "FAILURE")
        
        return recon_results
    
    async def run_vulnerability_analysis(self, recon_data: Dict) -> Dict[str, Any]:
        """Analyze vulnerabilities using CVE and ExploitDB"""
        self.console.print("\n🔍 VULNERABILITY ANALYSIS", style="bold cyan")
        self.console.print("═" * 30, style="cyan")
        
        vuln_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("🎯 Searching vulnerability databases...")
            
            try:
                # Use AI core to search for vulnerabilities
                vulnerabilities = await self.ai_core.search_vulnerabilities(recon_data)
                vuln_results = vulnerabilities
                
                self.log_mission_event("ANALYSIS", f"Found {len(vulnerabilities)} potential vulnerabilities", "SUCCESS")
                
                # Display found vulnerabilities
                if vulnerabilities:
                    table = Table(title="🎯 Discovered Vulnerabilities")
                    table.add_column("CVE/ID", style="red")
                    table.add_column("Description", style="yellow")
                    table.add_column("Severity", style="bold")
                    
                    for vuln in vulnerabilities[:5]:  # Show top 5
                        severity_color = "red" if vuln.get("severity") == "HIGH" else "yellow"
                        table.add_row(
                            vuln.get("id", "N/A"),
                            vuln.get("description", "")[:50] + "...",
                            vuln.get("severity", "UNKNOWN"),
                            style=severity_color
                        )
                    
                    self.console.print(table)
                
            except Exception as e:
                self.log_mission_event("ANALYSIS", f"Vulnerability analysis failed: {str(e)}", "FAILURE")
        
        return vuln_results
    
    async def run_web_attack_phase(self, recon_data: Dict, vuln_data: Dict) -> Dict[str, Any]:
        """Execute web application attacks"""
        self.print_ascii_banner("ATTACK")
        self.log_mission_event("ATTACK", "Launching AI-driven web attacks", "INFO")
        
        attack_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # SQL Injection attacks
            task1 = progress.add_task("💉 Testing SQL Injection...", total=100)
            try:
                sqli_results = await self.web_attack_engine.test_sql_injection(self.target_url)
                attack_results["sql_injection"] = sqli_results
                progress.update(task1, advance=100)
                if sqli_results.get("vulnerable"):
                    self.log_mission_event("ATTACK", "SQL Injection vulnerability found", "SUCCESS")
                else:
                    self.log_mission_event("ATTACK", "No SQL Injection vulnerabilities", "INFO")
            except Exception as e:
                progress.update(task1, advance=100)
                self.log_mission_event("ATTACK", f"SQL Injection test failed: {str(e)}", "FAILURE")
            
            # XSS attacks
            task2 = progress.add_task("🌐 Testing Cross-Site Scripting...", total=100)
            try:
                xss_results = await self.web_attack_engine.test_xss(self.target_url)
                attack_results["xss"] = xss_results
                progress.update(task2, advance=100)
                if xss_results.get("vulnerable"):
                    self.log_mission_event("ATTACK", "XSS vulnerability found", "SUCCESS")
                else:
                    self.log_mission_event("ATTACK", "No XSS vulnerabilities", "INFO")
            except Exception as e:
                progress.update(task2, advance=100)
                self.log_mission_event("ATTACK", f"XSS test failed: {str(e)}", "FAILURE")
            
            # SSRF attacks
            task3 = progress.add_task("🔄 Testing Server-Side Request Forgery...", total=100)
            try:
                ssrf_results = await self.web_attack_engine.test_ssrf(self.target_url)
                attack_results["ssrf"] = ssrf_results
                progress.update(task3, advance=100)
                if ssrf_results.get("vulnerable"):
                    self.log_mission_event("ATTACK", "SSRF vulnerability found", "SUCCESS")
                else:
                    self.log_mission_event("ATTACK", "No SSRF vulnerabilities", "INFO")
            except Exception as e:
                progress.update(task3, advance=100)
                self.log_mission_event("ATTACK", f"SSRF test failed: {str(e)}", "FAILURE")
            
            # SSTI attacks
            task4 = progress.add_task("🎭 Testing Server-Side Template Injection...", total=100)
            try:
                ssti_results = await self.web_attack_engine.test_ssti(self.target_url)
                attack_results["ssti"] = ssti_results
                progress.update(task4, advance=100)
                if ssti_results.get("vulnerable"):
                    self.log_mission_event("ATTACK", "SSTI vulnerability found", "SUCCESS")
                else:
                    self.log_mission_event("ATTACK", "No SSTI vulnerabilities", "INFO")
            except Exception as e:
                progress.update(task4, advance=100)
                self.log_mission_event("ATTACK", f"SSTI test failed: {str(e)}", "FAILURE")
        
        return attack_results
    
    async def run_database_exploitation(self, attack_results: Dict) -> Dict[str, Any]:
        """Execute database exploitation phase"""
        self.print_ascii_banner("DB_EXPLOIT")
        self.log_mission_event("DB_EXPLOIT", "Starting database exploitation", "INFO")
        
        db_results = {}
        
        # Only proceed if SQL injection was found
        if not attack_results.get("sql_injection", {}).get("vulnerable"):
            self.log_mission_event("DB_EXPLOIT", "Skipping - no SQL injection vector found", "INFO")
            return db_results
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Database fingerprinting
            task1 = progress.add_task("🔍 Fingerprinting database...", total=100)
            try:
                db_info = await self.db_exploit_engine.fingerprint_database(self.target_url)
                db_results["fingerprint"] = db_info
                progress.update(task1, advance=100)
                self.log_mission_event("DB_EXPLOIT", f"Database identified: {db_info.get('type')}", "SUCCESS")
            except Exception as e:
                progress.update(task1, advance=100)
                self.log_mission_event("DB_EXPLOIT", f"Database fingerprinting failed: {str(e)}", "FAILURE")
            
            # Table enumeration
            task2 = progress.add_task("📊 Enumerating tables...", total=100)
            try:
                tables = await self.db_exploit_engine.enumerate_tables(self.target_url)
                db_results["tables"] = tables
                progress.update(task2, advance=100)
                self.log_mission_event("DB_EXPLOIT", f"Found {len(tables)} tables", "SUCCESS")
            except Exception as e:
                progress.update(task2, advance=100)
                self.log_mission_event("DB_EXPLOIT", f"Table enumeration failed: {str(e)}", "FAILURE")
            
            # Data extraction (authorized only)
            if not self.safe_mode:
                task3 = progress.add_task("💾 Extracting sensitive data...", total=100)
                try:
                    sensitive_data = await self.db_exploit_engine.extract_sensitive_data(self.target_url)
                    db_results["sensitive_data"] = sensitive_data
                    progress.update(task3, advance=100)
                    self.log_mission_event("DB_EXPLOIT", "Sensitive data extracted", "SUCCESS")
                except Exception as e:
                    progress.update(task3, advance=100)
                    self.log_mission_event("DB_EXPLOIT", f"Data extraction failed: {str(e)}", "FAILURE")
        
        return db_results
    
    async def run_admin_login_attacks(self, recon_data: Dict) -> Dict[str, Any]:
        """Execute admin login and privilege escalation attacks"""
        self.print_ascii_banner("ADMIN_LOGIN")
        self.log_mission_event("ADMIN_LOGIN", "Starting admin portal attacks", "INFO")
        
        admin_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Admin portal discovery
            task1 = progress.add_task("🔍 Discovering admin portals...", total=100)
            try:
                admin_portals = await self.admin_login_engine.discover_admin_portals(self.target_url)
                admin_results["portals"] = admin_portals
                progress.update(task1, advance=100)
                self.log_mission_event("ADMIN_LOGIN", f"Found {len(admin_portals)} admin portals", "SUCCESS")
            except Exception as e:
                progress.update(task1, advance=100)
                self.log_mission_event("ADMIN_LOGIN", f"Portal discovery failed: {str(e)}", "FAILURE")
            
            # Credential brute force (if authorized)
            if not self.safe_mode and admin_portals:
                task2 = progress.add_task("🔓 Testing credentials...", total=100)
                try:
                    cred_results = await self.admin_login_engine.test_credentials(admin_portals[0])
                    admin_results["credentials"] = cred_results
                    progress.update(task2, advance=100)
                    if cred_results.get("success"):
                        self.log_mission_event("ADMIN_LOGIN", "Valid credentials found", "SUCCESS")
                    else:
                        self.log_mission_event("ADMIN_LOGIN", "No valid credentials found", "INFO")
                except Exception as e:
                    progress.update(task2, advance=100)
                    self.log_mission_event("ADMIN_LOGIN", f"Credential testing failed: {str(e)}", "FAILURE")
            
            # Session token analysis
            task3 = progress.add_task("🎫 Analyzing session tokens...", total=100)
            try:
                token_analysis = await self.admin_login_engine.analyze_session_tokens(self.target_url)
                admin_results["tokens"] = token_analysis
                progress.update(task3, advance=100)
                self.log_mission_event("ADMIN_LOGIN", "Session token analysis complete", "SUCCESS")
            except Exception as e:
                progress.update(task3, advance=100)
                self.log_mission_event("ADMIN_LOGIN", f"Token analysis failed: {str(e)}", "FAILURE")
        
        return admin_results
    
    async def ai_adaptive_chaining(self, all_results: Dict) -> Dict[str, Any]:
        """AI-driven adaptive attack chaining and retesting"""
        self.console.print("\n🤖 AI ADAPTIVE ATTACK CHAINING", style="bold magenta")
        self.console.print("═" * 35, style="magenta")
        
        adaptive_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("🧠 AI analyzing attack results...")
            
            try:
                # Let AI analyze results and suggest next steps
                ai_analysis = await self.ai_core.analyze_attack_results(all_results)
                adaptive_results["analysis"] = ai_analysis
                
                # Generate custom payloads based on findings
                custom_payloads = await self.ai_core.generate_custom_payloads(all_results)
                adaptive_results["custom_payloads"] = custom_payloads
                
                # Adaptive retesting of failed attacks
                retest_results = await self.ai_core.adaptive_retest(all_results, self.target_url)
                adaptive_results["retesting"] = retest_results
                
                self.log_mission_event("AI_CHAIN", "AI adaptive chaining complete", "SUCCESS")
                
            except Exception as e:
                self.log_mission_event("AI_CHAIN", f"AI chaining failed: {str(e)}", "FAILURE")
        
        return adaptive_results
    
    async def generate_mission_report(self, all_results: Dict, export_path: Optional[str] = None) -> str:
        """Generate comprehensive mission report"""
        self.console.print("\n📊 GENERATING MISSION REPORT", style="bold blue")
        self.console.print("═" * 30, style="blue")
        
        try:
            # Prepare report data
            report_data = {
                "mission_info": {
                    "target": self.target_url,
                    "start_time": self.start_time,
                    "end_time": datetime.now().isoformat(),
                    "duration": str(datetime.now() - self.start_time) if self.start_time else "Unknown",
                    "safe_mode": self.safe_mode
                },
                "results": all_results,
                "mission_log": self.mission_log,
                "ai_analysis": all_results.get("ai_adaptive", {})
            }
            
            # Generate reports
            report_paths = await self.report_generator.generate_one_click_report(report_data, export_path)
            
            self.log_mission_event("REPORT", f"Report generated: {report_paths}", "SUCCESS")
            
            # Display summary
            self.display_mission_summary(report_data)
            
            return report_paths
            
        except Exception as e:
            self.log_mission_event("REPORT", f"Report generation failed: {str(e)}", "FAILURE")
            return None
    
    def display_mission_summary(self, report_data: Dict):
        """Display mission summary in terminal"""
        summary_table = Table(title="🎯 MISSION SUMMARY", show_header=True, header_style="bold cyan")
        summary_table.add_column("Phase", style="yellow")
        summary_table.add_column("Status", justify="center")
        summary_table.add_column("Results", style="green")
        
        results = report_data["results"]
        
        # Count successes for each phase
        recon_success = len(results.get("reconnaissance", {}).get("subdomains", [])) > 0
        vuln_success = len(results.get("vulnerabilities", [])) > 0
        attack_success = any(v.get("vulnerable") for v in results.get("web_attacks", {}).values())
        db_success = bool(results.get("database", {}).get("fingerprint"))
        admin_success = len(results.get("admin_login", {}).get("portals", [])) > 0
        
        summary_table.add_row("Reconnaissance", "✓" if recon_success else "✗", 
                             f"{len(results.get('reconnaissance', {}).get('subdomains', []))} subdomains found")
        summary_table.add_row("Vulnerability Analysis", "✓" if vuln_success else "✗",
                             f"{len(results.get('vulnerabilities', []))} vulnerabilities")
        summary_table.add_row("Web Attacks", "✓" if attack_success else "✗",
                             f"{'Success' if attack_success else 'No vulnerabilities found'}")
        summary_table.add_row("Database Exploitation", "✓" if db_success else "✗",
                             f"{'Database accessed' if db_success else 'No access'}")
        summary_table.add_row("Admin Login", "✓" if admin_success else "✗",
                             f"{len(results.get('admin_login', {}).get('portals', []))} portals found")
        
        self.console.print(summary_table)
    
    async def execute_one_click_attack(self, target_url: str, safe_mode: bool = False, 
                                     verbose: bool = False, export_path: Optional[str] = None):
        """Main execution method for One-Click Attack"""
        self.target_url = target_url
        self.safe_mode = safe_mode
        self.verbose = verbose
        self.start_time = datetime.now()
        
        # Print main banner
        self.print_ascii_banner("MAIN")
        
        self.console.print(f"🎯 Target: {target_url}", style="bold white")
        self.console.print(f"🛡️ Safe Mode: {'ON' if safe_mode else 'OFF'}", 
                          style="green" if safe_mode else "red")
        self.console.print(f"📝 Verbose: {'ON' if verbose else 'OFF'}", style="cyan")
        self.console.print("=" * 80, style="white")
        
        # Step 1: Admin Authentication
        if not await self.authenticate_admin():
            self.console.print("\n❌ MISSION ABORTED - Authentication Required", style="bold red")
            return False
        
        try:
            # Step 2: Reconnaissance
            recon_results = await self.run_reconnaissance_phase()
            
            # Step 3: Vulnerability Analysis
            vuln_results = await self.run_vulnerability_analysis(recon_results)
            
            # Step 4: Web Application Attacks
            attack_results = await self.run_web_attack_phase(recon_results, vuln_results)
            
            # Step 5: Database Exploitation
            db_results = await self.run_database_exploitation(attack_results)
            
            # Step 6: Admin Login Testing
            admin_results = await self.run_admin_login_attacks(recon_results)
            
            # Step 7: AI Adaptive Chaining
            all_attack_results = {
                "reconnaissance": recon_results,
                "vulnerabilities": vuln_results,
                "web_attacks": attack_results,
                "database": db_results,
                "admin_login": admin_results
            }
            
            ai_results = await self.ai_adaptive_chaining(all_attack_results)
            all_attack_results["ai_adaptive"] = ai_results
            
            # Step 8: Generate Report
            report_path = await self.generate_mission_report(all_attack_results, export_path)
            
            # Mission Complete
            self.console.print("\n" + "="*80, style="green")
            self.console.print("🎉 MISSION COMPLETE - All attacks executed with AI adaptive chaining", 
                             style="bold green")
            if report_path:
                self.console.print(f"📋 Report saved: {report_path}", style="cyan")
            self.console.print("="*80, style="green")
            
            return True
            
        except Exception as e:
            self.log_mission_event("SYSTEM", f"Mission failed: {str(e)}", "CRITICAL")
            self.console.print(f"\n❌ MISSION FAILED: {str(e)}", style="bold red")
            return False

# Main function for testing
async def main():
    """Test function for One-Click Attack"""
    orchestrator = OneClickAttackOrchestrator()
    
    # Test with example target
    await orchestrator.execute_one_click_attack(
        target_url="https://demo.testfire.net",
        safe_mode=True,
        verbose=True,
        export_path="./reports/"
    )

if __name__ == "__main__":
    asyncio.run(main())