"""
Agent DS Reporting Engine
Automated report generation for penetration testing results
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64
import io
import logging

# Import reporting libraries
try:
    import jinja2
    from jinja2 import Environment, FileSystemLoader, Template
except ImportError:
    jinja2 = None

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
except ImportError:
    pass

try:
    import weasyprint
except ImportError:
    weasyprint = None

from core.config.settings import Config
from core.database.manager import DatabaseManager
from core.utils.logger import get_logger

logger = get_logger('reporting_engine')

class ReportEngine:
    """Main reporting engine for Agent DS"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        
        # Set up templates directory
        self.templates_dir = Path(__file__).parent / 'templates'
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        if jinja2:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=True
            )
        else:
            self.jinja_env = None
            logger.warning("Jinja2 not available - HTML reports disabled")
    
    async def generate_mission_report(self, mission_id: str, 
                                    report_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive mission report"""
        logger.info(f"Generating mission report for mission {mission_id}")
        
        if not report_config:
            report_config = {}
        
        # Gather mission data
        mission_data = await self._gather_mission_data(mission_id)
        
        # Generate different report formats
        reports = {}
        
        # Generate executive summary
        reports['executive_summary'] = await self._generate_executive_summary(mission_data)
        
        # Generate technical report
        if report_config.get('include_technical', True):
            reports['technical_report'] = await self._generate_technical_report(mission_data)
        
        # Generate PDF report
        if report_config.get('generate_pdf', True):
            pdf_path = await self._generate_pdf_report(mission_data, mission_id)
            reports['pdf_report'] = pdf_path
        
        # Generate HTML report
        if report_config.get('generate_html', True) and self.jinja_env:
            html_path = await self._generate_html_report(mission_data, mission_id)
            reports['html_report'] = html_path
        
        # Generate JSON export
        if report_config.get('generate_json', True):
            json_path = await self._generate_json_export(mission_data, mission_id)
            reports['json_export'] = json_path
        
        # Store report metadata in database
        report_metadata = {
            'mission_id': mission_id,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'report_types': list(reports.keys()),
            'vulnerability_count': len(mission_data.get('vulnerabilities', [])),
            'critical_findings': mission_data.get('risk_summary', {}).get('critical', 0),
            'high_findings': mission_data.get('risk_summary', {}).get('high', 0)
        }
        
        self.db_manager.store_report(
            mission_id=mission_id,
            report_type='comprehensive',
            report_data=report_metadata,
            file_path=reports.get('pdf_report'),
            metadata=reports
        )
        
        logger.info(f"Mission report generated successfully for {mission_id}")
        return reports
    
    async def _gather_mission_data(self, mission_id: str) -> Dict[str, Any]:
        """Gather all mission data for reporting"""
        mission_data = {
            'mission_id': mission_id,
            'generated_at': datetime.now(timezone.utc),
            'hosts': [],
            'services': [],
            'vulnerabilities': [],
            'attacks': [],
            'risk_summary': {},
            'recommendations': []
        }
        
        try:
            # Get mission info from database
            mission_info = self.db_manager.get_mission(mission_id)
            if mission_info:
                mission_data.update(mission_info)
            
            # Get reconnaissance results
            recon_results = self.db_manager.get_recon_results(mission_id)
            if recon_results:
                mission_data['hosts'] = recon_results.get('hosts', [])
                mission_data['services'] = recon_results.get('services', [])
            
            # Get vulnerability data
            vulnerabilities = self.db_manager.get_vulnerabilities(mission_id)
            mission_data['vulnerabilities'] = vulnerabilities or []
            
            # Get attack results
            attacks = self.db_manager.get_attack_attempts(mission_id)
            mission_data['attacks'] = attacks or []
            
            # Calculate risk summary
            mission_data['risk_summary'] = self._calculate_risk_summary(mission_data['vulnerabilities'])
            
            # Generate recommendations
            mission_data['recommendations'] = self._generate_recommendations(mission_data)
            
        except Exception as e:
            logger.error(f"Error gathering mission data: {str(e)}")
        
        return mission_data
    
    def _calculate_risk_summary(self, vulnerabilities: List[Dict]) -> Dict[str, int]:
        """Calculate risk level summary"""
        risk_summary = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'info').lower()
            if severity in risk_summary:
                risk_summary[severity] += 1
        
        return risk_summary
    
    def _generate_recommendations(self, mission_data: Dict) -> List[Dict]:
        """Generate security recommendations"""
        recommendations = []
        
        vulnerabilities = mission_data.get('vulnerabilities', [])
        risk_summary = mission_data.get('risk_summary', {})
        
        # Critical vulnerabilities recommendations
        if risk_summary.get('critical', 0) > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Immediate Action Required',
                'recommendation': 'Address all critical vulnerabilities immediately. These pose significant risk to organizational security.',
                'timeline': 'Within 24-48 hours'
            })
        
        # High severity recommendations
        if risk_summary.get('high', 0) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Security Hardening',
                'recommendation': 'Implement security patches and configuration changes for high-severity findings.',
                'timeline': 'Within 1-2 weeks'
            })
        
        # Common vulnerability patterns
        vuln_types = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get('type', 'unknown')
            vuln_types[vuln_type] = vuln_types.get(vuln_type, 0) + 1
        
        # SQL Injection recommendations
        if 'sql_injection' in vuln_types:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Input Validation',
                'recommendation': 'Implement parameterized queries and input validation to prevent SQL injection attacks.',
                'timeline': 'Within 1 week'
            })
        
        # XSS recommendations
        if 'xss' in vuln_types:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Output Encoding',
                'recommendation': 'Implement proper output encoding and Content Security Policy (CSP) headers.',
                'timeline': 'Within 2 weeks'
            })
        
        # Default credentials
        if any('default' in vuln.get('description', '').lower() for vuln in vulnerabilities):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Access Control',
                'recommendation': 'Change all default credentials and implement strong password policies.',
                'timeline': 'Immediately'
            })
        
        return recommendations
    
    async def _generate_executive_summary(self, mission_data: Dict) -> Dict[str, Any]:
        """Generate executive summary"""
        risk_summary = mission_data.get('risk_summary', {})
        total_vulns = sum(risk_summary.values())
        
        # Calculate risk score (weighted)
        risk_score = (
            risk_summary.get('critical', 0) * 10 +
            risk_summary.get('high', 0) * 7 +
            risk_summary.get('medium', 0) * 4 +
            risk_summary.get('low', 0) * 2 +
            risk_summary.get('info', 0) * 1
        )
        
        # Determine overall risk level
        if risk_summary.get('critical', 0) > 0:
            overall_risk = 'CRITICAL'
        elif risk_summary.get('high', 0) > 3:
            overall_risk = 'HIGH'
        elif risk_summary.get('high', 0) > 0 or risk_summary.get('medium', 0) > 5:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        return {
            'mission_id': mission_data.get('mission_id'),
            'assessment_date': mission_data.get('generated_at').strftime('%Y-%m-%d'),
            'total_hosts_scanned': len(mission_data.get('hosts', [])),
            'total_services_identified': len(mission_data.get('services', [])),
            'total_vulnerabilities': total_vulns,
            'risk_breakdown': risk_summary,
            'overall_risk_level': overall_risk,
            'risk_score': risk_score,
            'key_findings': self._extract_key_findings(mission_data),
            'immediate_actions': self._extract_immediate_actions(mission_data)
        }
    
    def _extract_key_findings(self, mission_data: Dict) -> List[str]:
        """Extract key findings for executive summary"""
        findings = []
        
        vulnerabilities = mission_data.get('vulnerabilities', [])
        risk_summary = mission_data.get('risk_summary', {})
        
        # Critical findings
        critical_vulns = [v for v in vulnerabilities if v.get('severity', '').lower() == 'critical']
        for vuln in critical_vulns[:3]:  # Top 3 critical
            findings.append(f"Critical vulnerability: {vuln.get('title', 'Unknown')}")
        
        # High-level patterns
        if risk_summary.get('critical', 0) > 0:
            findings.append(f"{risk_summary['critical']} critical vulnerabilities require immediate attention")
        
        if risk_summary.get('high', 0) > 5:
            findings.append(f"High number of high-severity vulnerabilities ({risk_summary['high']}) indicates systemic security issues")
        
        return findings[:5]  # Limit to 5 key findings
    
    def _extract_immediate_actions(self, mission_data: Dict) -> List[str]:
        """Extract immediate actions for executive summary"""
        actions = []
        
        recommendations = mission_data.get('recommendations', [])
        
        # Get critical and high priority recommendations
        for rec in recommendations:
            if rec.get('priority') in ['CRITICAL', 'HIGH']:
                actions.append(rec.get('recommendation'))
        
        return actions[:3]  # Limit to 3 immediate actions
    
    async def _generate_technical_report(self, mission_data: Dict) -> Dict[str, Any]:
        """Generate detailed technical report"""
        return {
            'methodology': self._get_testing_methodology(),
            'scope': {
                'hosts_tested': len(mission_data.get('hosts', [])),
                'services_identified': len(mission_data.get('services', [])),
                'test_duration': self._calculate_test_duration(mission_data)
            },
            'reconnaissance_results': self._format_recon_results(mission_data),
            'vulnerability_details': self._format_vulnerability_details(mission_data),
            'attack_results': self._format_attack_results(mission_data),
            'tool_output': self._format_tool_output(mission_data),
            'appendices': self._generate_appendices(mission_data)
        }
    
    def _get_testing_methodology(self) -> Dict[str, Any]:
        """Get testing methodology description"""
        return {
            'approach': 'Automated penetration testing using Agent DS framework',
            'phases': [
                'Reconnaissance and Information Gathering',
                'Vulnerability Identification and Analysis',
                'Exploitation and Attack Execution',
                'Post-Exploitation and Impact Assessment',
                'Reporting and Remediation Guidance'
            ],
            'tools_used': [
                'Nmap - Network discovery and service enumeration',
                'Masscan - High-speed port scanning',
                'SQLMap - SQL injection testing',
                'Metasploit - Exploitation framework',
                'Hydra - Brute force attacks',
                'OWASP ZAP - Web application security testing',
                'Custom AI-generated payloads'
            ],
            'standards_followed': [
                'OWASP Testing Guide',
                'NIST SP 800-115',
                'PTES (Penetration Testing Execution Standard)'
            ]
        }
    
    def _calculate_test_duration(self, mission_data: Dict) -> str:
        """Calculate test duration"""
        # This would calculate actual duration from mission timestamps
        return "Automated testing completed in under 1 hour"
    
    def _format_recon_results(self, mission_data: Dict) -> Dict[str, Any]:
        """Format reconnaissance results"""
        return {
            'hosts_discovered': mission_data.get('hosts', []),
            'services_identified': mission_data.get('services', []),
            'open_ports_summary': self._summarize_open_ports(mission_data),
            'service_versions': self._extract_service_versions(mission_data)
        }
    
    def _summarize_open_ports(self, mission_data: Dict) -> Dict[str, int]:
        """Summarize open ports"""
        port_summary = {}
        services = mission_data.get('services', [])
        
        for service in services:
            port = service.get('port')
            if port:
                port_summary[str(port)] = port_summary.get(str(port), 0) + 1
        
        return port_summary
    
    def _extract_service_versions(self, mission_data: Dict) -> List[Dict]:
        """Extract service version information"""
        versions = []
        services = mission_data.get('services', [])
        
        for service in services:
            if service.get('version'):
                versions.append({
                    'host': service.get('host'),
                    'port': service.get('port'),
                    'service': service.get('service'),
                    'version': service.get('version')
                })
        
        return versions
    
    def _format_vulnerability_details(self, mission_data: Dict) -> List[Dict]:
        """Format vulnerability details for technical report"""
        vulns = mission_data.get('vulnerabilities', [])
        formatted_vulns = []
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        sorted_vulns = sorted(vulns, key=lambda x: severity_order.get(x.get('severity', 'info').lower(), 5))
        
        for vuln in sorted_vulns:
            formatted_vulns.append({
                'id': vuln.get('id'),
                'title': vuln.get('title'),
                'severity': vuln.get('severity'),
                'cvss_score': vuln.get('cvss_score'),
                'description': vuln.get('description'),
                'affected_hosts': vuln.get('affected_hosts', []),
                'proof_of_concept': vuln.get('proof_of_concept'),
                'remediation': vuln.get('remediation'),
                'references': vuln.get('references', [])
            })
        
        return formatted_vulns
    
    def _format_attack_results(self, mission_data: Dict) -> List[Dict]:
        """Format attack execution results"""
        attacks = mission_data.get('attacks', [])
        formatted_attacks = []
        
        for attack in attacks:
            formatted_attacks.append({
                'target': attack.get('target'),
                'attack_type': attack.get('attack_type'),
                'tool_used': attack.get('tool_name'),
                'success': attack.get('success'),
                'timestamp': attack.get('timestamp'),
                'details': attack.get('result_data', {})
            })
        
        return formatted_attacks
    
    def _format_tool_output(self, mission_data: Dict) -> Dict[str, Any]:
        """Format tool output for appendix"""
        return {
            'reconnaissance_output': 'Detailed tool output available in raw logs',
            'vulnerability_scan_output': 'Vulnerability scanner results archived',
            'exploitation_attempts': 'Attack tool output and payloads documented'
        }
    
    def _generate_appendices(self, mission_data: Dict) -> Dict[str, Any]:
        """Generate report appendices"""
        return {
            'vulnerability_references': self._compile_vuln_references(mission_data),
            'remediation_resources': self._compile_remediation_resources(),
            'tool_configurations': self._get_tool_configurations(),
            'raw_scan_data': 'Available upon request'
        }
    
    def _compile_vuln_references(self, mission_data: Dict) -> List[str]:
        """Compile vulnerability references"""
        references = set()
        vulnerabilities = mission_data.get('vulnerabilities', [])
        
        for vuln in vulnerabilities:
            vuln_refs = vuln.get('references', [])
            references.update(vuln_refs)
        
        return sorted(list(references))
    
    def _compile_remediation_resources(self) -> List[Dict]:
        """Compile remediation resources"""
        return [
            {
                'title': 'OWASP Top 10',
                'url': 'https://owasp.org/www-project-top-ten/',
                'description': 'Most critical web application security risks'
            },
            {
                'title': 'SANS Top 25',
                'url': 'https://www.sans.org/top25-software-errors/',
                'description': 'Most dangerous software errors'
            },
            {
                'title': 'NIST Cybersecurity Framework',
                'url': 'https://www.nist.gov/cyberframework',
                'description': 'Framework for improving critical infrastructure cybersecurity'
            }
        ]
    
    def _get_tool_configurations(self) -> Dict[str, str]:
        """Get tool configuration details"""
        return {
            'nmap': 'Standard TCP SYN scan with service detection',
            'sqlmap': 'Batch mode with medium risk and level 3 detection',
            'metasploit': 'Framework 6.x with standard auxiliary modules',
            'hydra': 'Dictionary attack with common credentials'
        }
    
    async def _generate_pdf_report(self, mission_data: Dict, mission_id: str) -> Optional[str]:
        """Generate PDF report using ReportLab"""
        try:
            # Create reports directory
            reports_dir = Path(self.config.get('reporting.output_directory', 'reports'))
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"agent_ds_report_{mission_id}_{timestamp}.pdf"
            filepath = reports_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            
            # Title page
            story.append(Paragraph("Agent DS Penetration Testing Report", title_style))
            story.append(Spacer(1, 0.5*inch))
            
            # Executive summary
            exec_summary = await self._generate_executive_summary(mission_data)
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            story.append(Paragraph(f"Mission ID: {mission_id}", styles['Normal']))
            story.append(Paragraph(f"Assessment Date: {exec_summary['assessment_date']}", styles['Normal']))
            story.append(Paragraph(f"Overall Risk Level: {exec_summary['overall_risk_level']}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Risk summary table
            risk_data = [
                ['Risk Level', 'Count'],
                ['Critical', str(exec_summary['risk_breakdown'].get('critical', 0))],
                ['High', str(exec_summary['risk_breakdown'].get('high', 0))],
                ['Medium', str(exec_summary['risk_breakdown'].get('medium', 0))],
                ['Low', str(exec_summary['risk_breakdown'].get('low', 0))],
                ['Info', str(exec_summary['risk_breakdown'].get('info', 0))]
            ]
            
            risk_table = Table(risk_data)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(risk_table)
            story.append(PageBreak())
            
            # Technical details
            story.append(Paragraph("Technical Findings", styles['Heading2']))
            
            vulnerabilities = mission_data.get('vulnerabilities', [])
            if vulnerabilities:
                for i, vuln in enumerate(vulnerabilities[:10]):  # Limit to top 10
                    story.append(Paragraph(f"Finding {i+1}: {vuln.get('title', 'Unknown')}", styles['Heading3']))
                    story.append(Paragraph(f"Severity: {vuln.get('severity', 'Unknown')}", styles['Normal']))
                    story.append(Paragraph(f"Description: {vuln.get('description', 'No description available')}", styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return None
    
    async def _generate_html_report(self, mission_data: Dict, mission_id: str) -> Optional[str]:
        """Generate HTML report using Jinja2 templates"""
        if not self.jinja_env:
            logger.warning("Jinja2 not available - cannot generate HTML report")
            return None
        
        try:
            # Create HTML template if it doesn't exist
            template_path = self.templates_dir / 'mission_report.html'
            if not template_path.exists():
                await self._create_html_template(template_path)
            
            # Load template
            template = self.jinja_env.get_template('mission_report.html')
            
            # Prepare template data
            template_data = {
                'mission_data': mission_data,
                'executive_summary': await self._generate_executive_summary(mission_data),
                'technical_report': await self._generate_technical_report(mission_data),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'mission_id': mission_id
            }
            
            # Render HTML
            html_content = template.render(**template_data)
            
            # Save HTML file
            reports_dir = Path(self.config.get('reporting.output_directory', 'reports'))
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"agent_ds_report_{mission_id}_{timestamp}.html"
            filepath = reports_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return None
    
    async def _create_html_template(self, template_path: Path):
        """Create basic HTML template"""
        template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent DS Penetration Testing Report - {{ mission_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .risk-critical { color: #d32f2f; font-weight: bold; }
        .risk-high { color: #f57c00; font-weight: bold; }
        .risk-medium { color: #fbc02d; font-weight: bold; }
        .risk-low { color: #388e3c; font-weight: bold; }
        .risk-info { color: #1976d2; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f5f5f5; }
        .vuln-title { font-weight: bold; margin-top: 20px; }
        .footer { margin-top: 50px; text-align: center; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Agent DS Penetration Testing Report</h1>
        <p>Mission ID: {{ mission_id }}</p>
        <p>Generated: {{ generated_at }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p><strong>Overall Risk Level:</strong> 
            <span class="risk-{{ executive_summary.overall_risk_level.lower() }}">
                {{ executive_summary.overall_risk_level }}
            </span>
        </p>
        <p><strong>Total Vulnerabilities Found:</strong> {{ executive_summary.total_vulnerabilities }}</p>
        
        <h3>Risk Breakdown</h3>
        <table>
            <tr>
                <th>Risk Level</th>
                <th>Count</th>
            </tr>
            {% for level, count in executive_summary.risk_breakdown.items() %}
            <tr>
                <td class="risk-{{ level }}">{{ level.title() }}</td>
                <td>{{ count }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>Key Findings</h2>
        <ul>
        {% for finding in executive_summary.key_findings %}
            <li>{{ finding }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="section">
        <h2>Immediate Actions Required</h2>
        <ol>
        {% for action in executive_summary.immediate_actions %}
            <li>{{ action }}</li>
        {% endfor %}
        </ol>
    </div>

    <div class="section">
        <h2>Technical Findings</h2>
        {% for vuln in mission_data.vulnerabilities[:10] %}
        <div class="vuln-title">
            Finding {{ loop.index }}: {{ vuln.title }}
        </div>
        <p><strong>Severity:</strong> 
            <span class="risk-{{ vuln.severity.lower() }}">{{ vuln.severity }}</span>
        </p>
        <p><strong>Description:</strong> {{ vuln.description }}</p>
        {% if vuln.remediation %}
        <p><strong>Remediation:</strong> {{ vuln.remediation }}</p>
        {% endif %}
        <hr>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        {% for rec in mission_data.recommendations %}
        <h3>{{ rec.priority }} Priority: {{ rec.category }}</h3>
        <p>{{ rec.recommendation }}</p>
        <p><strong>Timeline:</strong> {{ rec.timeline }}</p>
        {% endfor %}
    </div>

    <div class="footer">
        <p>This report was generated by Agent DS - Autonomous Red-Team CLI AI Framework</p>
        <p>For questions or clarifications, please contact your security team.</p>
    </div>
</body>
</html>"""
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
    
    async def _generate_json_export(self, mission_data: Dict, mission_id: str) -> Optional[str]:
        """Generate JSON export of mission data"""
        try:
            # Create reports directory
            reports_dir = Path(self.config.get('reporting.output_directory', 'reports'))
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"agent_ds_data_{mission_id}_{timestamp}.json"
            filepath = reports_dir / filename
            
            # Prepare export data
            export_data = {
                'export_info': {
                    'mission_id': mission_id,
                    'exported_at': datetime.now(timezone.utc).isoformat(),
                    'agent_ds_version': '1.0.0'
                },
                'executive_summary': await self._generate_executive_summary(mission_data),
                'technical_report': await self._generate_technical_report(mission_data),
                'raw_data': mission_data
            }
            
            # Write JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"JSON export generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating JSON export: {str(e)}")
            return None