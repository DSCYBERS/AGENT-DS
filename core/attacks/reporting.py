#!/usr/bin/env python3
"""
Agent DS - Comprehensive Reporting System
Advanced report generation with multiple formats and executive dashboards

This module provides comprehensive reporting capabilities including:
- PDF reports with charts and visualizations
- JSON reports for API integration
- HTML reports with interactive elements
- Text reports for command-line usage
- Executive dashboards and summaries
- Vulnerability severity analysis
- Attack timeline visualization

Author: Agent DS Team
Version: 2.0
Date: September 16, 2025
"""

import asyncio
import json
import os
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Chart generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import seaborn as sns
    import numpy as np
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

# HTML template engine
try:
    from jinja2 import Environment, BaseLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

class ReportGenerator:
    """
    Comprehensive report generator for Agent DS attack results
    """
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.report_id = hashlib.md5(str(self.timestamp).encode()).hexdigest()[:8]
        
        # Report templates
        self.html_template = self._get_html_template()
        self.css_styles = self._get_css_styles()
        
        # Chart settings
        if CHARTS_AVAILABLE:
            plt.style.use('dark_background')
            sns.set_palette("husl")
    
    async def generate_json_report(self, attack_results: Dict[str, Any], output_path: Path):
        """Generate JSON report"""
        
        report_data = {
            'metadata': self._generate_metadata(attack_results),
            'executive_summary': self._generate_executive_summary(attack_results),
            'attack_timeline': self._generate_attack_timeline(attack_results),
            'vulnerability_details': self._extract_vulnerability_details(attack_results),
            'severity_analysis': self._analyze_severity_distribution(attack_results),
            'recommendations': self._generate_recommendations(attack_results),
            'technical_details': self._extract_technical_details(attack_results),
            'appendices': self._generate_appendices(attack_results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    async def generate_pdf_report(self, attack_results: Dict[str, Any], output_path: Path):
        """Generate PDF report"""
        
        if not PDF_AVAILABLE:
            # Fallback to text report
            await self.generate_text_report(attack_results, output_path.with_suffix('.txt'))
            return
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkred
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Title page
        story.append(Paragraph("AGENT DS - PENETRATION TEST REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        metadata = self._generate_metadata(attack_results)
        story.append(Paragraph("Report Information", heading_style))
        
        meta_data = [
            ['Report ID:', self.report_id],
            ['Generated:', metadata['timestamp']],
            ['Target:', metadata['target']],
            ['Attack Type:', metadata['attack_type']],
            ['Duration:', metadata['duration']],
            ['Agent Version:', metadata['agent_version']]
        ]
        
        meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(meta_table)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        summary = self._generate_executive_summary(attack_results)
        
        for paragraph in summary['key_points']:
            story.append(Paragraph(f"• {paragraph}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Risk Rating
        risk_level = summary['overall_risk_level']
        risk_color = {
            'CRITICAL': colors.red,
            'HIGH': colors.orange,
            'MEDIUM': colors.yellow,
            'LOW': colors.green
        }.get(risk_level, colors.grey)
        
        risk_style = ParagraphStyle(
            'RiskStyle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=risk_color,
            fontName='Helvetica-Bold'
        )
        
        story.append(Paragraph(f"Overall Risk Level: {risk_level}", risk_style))
        story.append(Spacer(1, 20))
        
        # Vulnerability Summary
        story.append(Paragraph("Vulnerability Summary", heading_style))
        
        vulnerabilities = self._extract_vulnerability_details(attack_results)
        if vulnerabilities:
            vuln_data = [['ID', 'Type', 'Severity', 'Description']]
            
            for i, vuln in enumerate(vulnerabilities[:10]):  # Limit for PDF
                vuln_data.append([
                    f"V-{i+1:03d}",
                    vuln.get('type', 'Unknown'),
                    vuln.get('severity', 'Unknown'),
                    vuln.get('description', 'No description')[:50] + "..."
                ])
            
            vuln_table = Table(vuln_data, colWidths=[0.8*inch, 1.2*inch, 1*inch, 3*inch])
            vuln_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(vuln_table)
        else:
            story.append(Paragraph("No vulnerabilities detected.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", heading_style))
        recommendations = self._generate_recommendations(attack_results)
        
        for i, rec in enumerate(recommendations['immediate_actions'][:5]):
            story.append(Paragraph(f"{i+1}. {rec}", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Technical Details
        story.append(Paragraph("Technical Details", heading_style))
        technical = self._extract_technical_details(attack_results)
        
        for section, details in technical.items():
            if isinstance(details, dict) and details:
                story.append(Paragraph(f"{section.replace('_', ' ').title()}:", styles['Heading3']))
                
                if isinstance(details, dict):
                    for key, value in list(details.items())[:5]:  # Limit entries
                        story.append(Paragraph(f"• {key}: {value}", styles['Normal']))
                
                story.append(Spacer(1, 8))
        
        # Build PDF
        doc.build(story)
    
    async def generate_html_report(self, attack_results: Dict[str, Any], output_path: Path):
        """Generate HTML report with interactive elements"""
        
        # Generate data for template
        report_data = {
            'metadata': self._generate_metadata(attack_results),
            'executive_summary': self._generate_executive_summary(attack_results),
            'vulnerability_details': self._extract_vulnerability_details(attack_results),
            'severity_analysis': self._analyze_severity_distribution(attack_results),
            'attack_timeline': self._generate_attack_timeline(attack_results),
            'recommendations': self._generate_recommendations(attack_results),
            'technical_details': self._extract_technical_details(attack_results),
            'charts': await self._generate_charts(attack_results) if CHARTS_AVAILABLE else {},
            'css_styles': self.css_styles
        }
        
        # Generate HTML
        if JINJA2_AVAILABLE:
            env = Environment(loader=BaseLoader())
            template = env.from_string(self.html_template)
            html_content = template.render(**report_data)
        else:
            html_content = self._generate_simple_html(report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    async def generate_text_report(self, attack_results: Dict[str, Any], output_path: Path):
        """Generate plain text report"""
        
        lines = []
        
        # Header
        lines.extend([
            "=" * 80,
            "AGENT DS - PENETRATION TEST REPORT",
            "=" * 80,
            ""
        ])
        
        # Metadata
        metadata = self._generate_metadata(attack_results)
        lines.extend([
            "REPORT INFORMATION",
            "-" * 20,
            f"Report ID: {self.report_id}",
            f"Generated: {metadata['timestamp']}",
            f"Target: {metadata['target']}",
            f"Attack Type: {metadata['attack_type']}",
            f"Duration: {metadata['duration']}",
            f"Agent Version: {metadata['agent_version']}",
            ""
        ])
        
        # Executive Summary
        summary = self._generate_executive_summary(attack_results)
        lines.extend([
            "EXECUTIVE SUMMARY",
            "-" * 17,
            f"Overall Risk Level: {summary['overall_risk_level']}",
            f"Vulnerabilities Found: {summary['total_vulnerabilities']}",
            f"Critical Issues: {summary['critical_vulnerabilities']}",
            "",
            "Key Findings:",
        ])
        
        for point in summary['key_points']:
            lines.append(f"  • {point}")
        
        lines.append("")
        
        # Vulnerabilities
        vulnerabilities = self._extract_vulnerability_details(attack_results)
        lines.extend([
            "VULNERABILITY DETAILS",
            "-" * 21,
            ""
        ])
        
        if vulnerabilities:
            for i, vuln in enumerate(vulnerabilities):
                lines.extend([
                    f"Vulnerability #{i+1}",
                    f"  Type: {vuln.get('type', 'Unknown')}",
                    f"  Severity: {vuln.get('severity', 'Unknown')}",
                    f"  Confidence: {vuln.get('confidence', 'Unknown')}",
                    f"  Description: {vuln.get('description', 'No description')}",
                    f"  Location: {vuln.get('location', 'Not specified')}",
                ])
                
                if vuln.get('payload'):
                    lines.append(f"  Payload: {vuln['payload']}")
                
                lines.append("")
        else:
            lines.append("No vulnerabilities detected.")
        
        lines.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations(attack_results)
        lines.extend([
            "RECOMMENDATIONS",
            "-" * 15,
            "",
            "Immediate Actions:",
        ])
        
        for i, action in enumerate(recommendations['immediate_actions']):
            lines.append(f"  {i+1}. {action}")
        
        lines.extend([
            "",
            "Long-term Improvements:",
        ])
        
        for i, improvement in enumerate(recommendations['long_term_improvements']):
            lines.append(f"  {i+1}. {improvement}")
        
        lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _generate_metadata(self, attack_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate report metadata"""
        
        start_time = attack_results.get('start_time', self.timestamp)
        end_time = attack_results.get('end_time', self.timestamp)
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        duration = end_time - start_time
        
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'target': attack_results.get('target_url', 'Unknown'),
            'attack_type': attack_results.get('attack_type', 'One-Click Attack'),
            'duration': str(duration).split('.')[0],  # Remove microseconds
            'agent_version': '2.0',
            'operator': 'Agent DS Autonomous System'
        }
    
    def _generate_executive_summary(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        
        # Count vulnerabilities by severity
        vulnerability_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'total': 0
        }
        
        key_findings = []
        
        # Analyze results from each attack phase
        for phase, results in attack_results.items():
            if isinstance(results, dict) and 'vulnerabilities' in results:
                vulns = results['vulnerabilities']
                vulnerability_counts['total'] += len(vulns)
                
                for vuln in vulns:
                    severity = vuln.get('severity', '').lower()
                    if 'critical' in severity:
                        vulnerability_counts['critical'] += 1
                    elif 'high' in severity:
                        vulnerability_counts['high'] += 1
                    elif 'medium' in severity:
                        vulnerability_counts['medium'] += 1
                    elif 'low' in severity:
                        vulnerability_counts['low'] += 1
        
        # Determine overall risk level
        if vulnerability_counts['critical'] > 0:
            overall_risk = 'CRITICAL'
        elif vulnerability_counts['high'] > 0:
            overall_risk = 'HIGH'
        elif vulnerability_counts['medium'] > 0:
            overall_risk = 'MEDIUM'
        elif vulnerability_counts['low'] > 0:
            overall_risk = 'LOW'
        else:
            overall_risk = 'MINIMAL'
        
        # Generate key findings
        if vulnerability_counts['total'] == 0:
            key_findings.append("No significant vulnerabilities were identified during the assessment.")
            key_findings.append("The target demonstrates good security posture.")
        else:
            key_findings.append(f"Assessment identified {vulnerability_counts['total']} vulnerabilities across multiple attack vectors.")
            
            if vulnerability_counts['critical'] > 0:
                key_findings.append(f"{vulnerability_counts['critical']} critical vulnerabilities require immediate attention.")
            
            if vulnerability_counts['high'] > 0:
                key_findings.append(f"{vulnerability_counts['high']} high-severity issues should be addressed promptly.")
            
            # Add specific findings based on attack results
            if attack_results.get('web_attacks', {}).get('sql_injection', {}).get('vulnerable'):
                key_findings.append("SQL injection vulnerabilities detected - data breach risk.")
            
            if attack_results.get('admin_login', {}).get('credentials'):
                key_findings.append("Weak administrative credentials discovered.")
            
            if attack_results.get('database', {}).get('accessible'):
                key_findings.append("Database access achieved - sensitive data at risk.")
        
        return {
            'overall_risk_level': overall_risk,
            'total_vulnerabilities': vulnerability_counts['total'],
            'critical_vulnerabilities': vulnerability_counts['critical'],
            'high_vulnerabilities': vulnerability_counts['high'],
            'medium_vulnerabilities': vulnerability_counts['medium'],
            'low_vulnerabilities': vulnerability_counts['low'],
            'key_points': key_findings
        }
    
    def _generate_attack_timeline(self, attack_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate attack timeline"""
        
        timeline = []
        base_time = datetime.now() - timedelta(hours=1)
        
        phases = ['reconnaissance', 'web_attacks', 'database', 'admin_login']
        
        for i, phase in enumerate(phases):
            if phase in attack_results:
                timeline.append({
                    'time': (base_time + timedelta(minutes=i*15)).strftime('%H:%M:%S'),
                    'phase': phase.replace('_', ' ').title(),
                    'status': 'Completed',
                    'findings': len(attack_results[phase].get('vulnerabilities', [])) if isinstance(attack_results[phase], dict) else 0
                })
        
        return timeline
    
    def _extract_vulnerability_details(self, attack_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract detailed vulnerability information"""
        
        vulnerabilities = []
        
        for phase, results in attack_results.items():
            if isinstance(results, dict) and 'vulnerabilities' in results:
                for vuln in results['vulnerabilities']:
                    vuln_detail = {
                        'id': f"V-{len(vulnerabilities)+1:03d}",
                        'phase': phase.replace('_', ' ').title(),
                        'type': vuln.get('type', 'Unknown'),
                        'severity': vuln.get('severity', 'Unknown'),
                        'confidence': vuln.get('confidence', 'Medium'),
                        'description': vuln.get('description', 'No description available'),
                        'location': vuln.get('url', vuln.get('location', 'Not specified')),
                        'payload': vuln.get('payload', ''),
                        'impact': vuln.get('impact', 'Not assessed'),
                        'remediation': vuln.get('remediation', 'Contact security team')
                    }
                    vulnerabilities.append(vuln_detail)
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        vulnerabilities.sort(key=lambda x: severity_order.get(x['severity'].upper(), 4))
        
        return vulnerabilities
    
    def _analyze_severity_distribution(self, attack_results: Dict[str, Any]) -> Dict[str, int]:
        """Analyze vulnerability severity distribution"""
        
        distribution = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'UNKNOWN': 0
        }
        
        vulnerabilities = self._extract_vulnerability_details(attack_results)
        
        for vuln in vulnerabilities:
            severity = vuln['severity'].upper()
            if severity in distribution:
                distribution[severity] += 1
            else:
                distribution['UNKNOWN'] += 1
        
        return distribution
    
    def _generate_recommendations(self, attack_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate security recommendations"""
        
        immediate_actions = []
        long_term_improvements = []
        
        vulnerabilities = self._extract_vulnerability_details(attack_results)
        
        # Immediate actions based on findings
        critical_vulns = [v for v in vulnerabilities if v['severity'].upper() == 'CRITICAL']
        if critical_vulns:
            immediate_actions.append("Address all critical vulnerabilities immediately to prevent data breaches")
            immediate_actions.append("Implement emergency patches for critical security flaws")
        
        # SQL injection specific
        sql_vulns = [v for v in vulnerabilities if 'sql' in v['type'].lower()]
        if sql_vulns:
            immediate_actions.append("Implement parameterized queries to prevent SQL injection attacks")
            immediate_actions.append("Review and sanitize all user input validation")
        
        # XSS specific
        xss_vulns = [v for v in vulnerabilities if 'xss' in v['type'].lower()]
        if xss_vulns:
            immediate_actions.append("Implement Content Security Policy (CSP) headers")
            immediate_actions.append("Encode all user output to prevent XSS attacks")
        
        # Admin access specific
        if attack_results.get('admin_login', {}).get('credentials'):
            immediate_actions.append("Reset all administrative passwords immediately")
            immediate_actions.append("Implement multi-factor authentication for admin accounts")
        
        # General recommendations
        if not immediate_actions:
            immediate_actions.append("Continue monitoring for new vulnerabilities")
            immediate_actions.append("Review security configurations regularly")
        
        # Long-term improvements
        long_term_improvements.extend([
            "Implement a comprehensive vulnerability management program",
            "Conduct regular penetration testing and security assessments",
            "Establish a security incident response plan",
            "Provide security awareness training for development teams",
            "Implement automated security scanning in CI/CD pipelines",
            "Establish security code review processes",
            "Deploy Web Application Firewall (WAF) protection",
            "Implement network segmentation and access controls"
        ])
        
        return {
            'immediate_actions': immediate_actions[:10],  # Limit to top 10
            'long_term_improvements': long_term_improvements[:10]
        }
    
    def _extract_technical_details(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical details for appendix"""
        
        technical = {}
        
        # Reconnaissance details
        if 'reconnaissance' in attack_results:
            recon = attack_results['reconnaissance']
            technical['reconnaissance'] = {
                'subdomains_found': len(recon.get('subdomains', [])),
                'ports_scanned': len(recon.get('ports', [])),
                'technologies_identified': recon.get('technologies', []),
                'dns_records': recon.get('dns_records', {}),
                'ssl_info': recon.get('ssl_info', {})
            }
        
        # Web attack details
        if 'web_attacks' in attack_results:
            web = attack_results['web_attacks']
            technical['web_attacks'] = {}
            
            for attack_type, results in web.items():
                if isinstance(results, dict):
                    technical['web_attacks'][attack_type] = {
                        'tested': True,
                        'vulnerable': results.get('vulnerable', False),
                        'payloads_tested': len(results.get('payloads_tested', [])),
                        'response_times': results.get('response_times', [])
                    }
        
        # Database details
        if 'database' in attack_results:
            db = attack_results['database']
            technical['database'] = {
                'fingerprint': db.get('fingerprint', {}),
                'tables_found': len(db.get('tables', [])),
                'data_extracted': bool(db.get('data')),
                'injection_points': db.get('injection_points', [])
            }
        
        return technical
    
    def _generate_appendices(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report appendices"""
        
        return {
            'payload_library': self._extract_payloads(attack_results),
            'raw_responses': self._extract_responses(attack_results),
            'tool_versions': {
                'agent_ds': '2.0',
                'python_version': '3.x',
                'scan_date': self.timestamp.isoformat()
            }
        }
    
    def _extract_payloads(self, attack_results: Dict[str, Any]) -> List[str]:
        """Extract all payloads used during the attack"""
        
        payloads = []
        
        for phase, results in attack_results.items():
            if isinstance(results, dict) and 'vulnerabilities' in results:
                for vuln in results['vulnerabilities']:
                    if vuln.get('payload'):
                        payloads.append(vuln['payload'])
        
        return list(set(payloads))  # Remove duplicates
    
    def _extract_responses(self, attack_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key HTTP responses"""
        
        responses = []
        
        for phase, results in attack_results.items():
            if isinstance(results, dict) and 'http_responses' in results:
                for response in results['http_responses']:
                    responses.append({
                        'url': response.get('url', ''),
                        'status_code': response.get('status_code', ''),
                        'content_type': response.get('content_type', ''),
                        'response_size': response.get('content_length', '')
                    })
        
        return responses[:20]  # Limit to 20 responses
    
    async def _generate_charts(self, attack_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate charts for HTML report"""
        
        if not CHARTS_AVAILABLE:
            return {}
        
        charts = {}
        
        # Severity distribution pie chart
        severity_dist = self._analyze_severity_distribution(attack_results)
        if sum(severity_dist.values()) > 0:
            charts['severity_pie'] = await self._create_severity_pie_chart(severity_dist)
        
        # Timeline chart
        timeline = self._generate_attack_timeline(attack_results)
        if timeline:
            charts['timeline_bar'] = await self._create_timeline_chart(timeline)
        
        return charts
    
    async def _create_severity_pie_chart(self, severity_dist: Dict[str, int]) -> str:
        """Create severity distribution pie chart"""
        
        # Filter out zero values
        filtered_dist = {k: v for k, v in severity_dist.items() if v > 0}
        
        if not filtered_dist:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#FF0000', '#FF8000', '#FFFF00', '#00FF00', '#808080']
        
        wedges, texts, autotexts = ax.pie(
            filtered_dist.values(),
            labels=filtered_dist.keys(),
            colors=colors[:len(filtered_dist)],
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title('Vulnerability Severity Distribution', fontsize=16, color='white')
        
        # Convert to base64
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img_data = canvas.buffer_rgba()
        img_array = np.asarray(img_data)
        
        # Save to bytes
        import io
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', facecolor='black', edgecolor='white')
        img_bytes.seek(0)
        
        img_base64 = base64.b64encode(img_bytes.read()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"
    
    async def _create_timeline_chart(self, timeline: List[Dict[str, str]]) -> str:
        """Create attack timeline chart"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        phases = [item['phase'] for item in timeline]
        findings = [int(item['findings']) for item in timeline]
        
        bars = ax.bar(phases, findings, color='cyan', alpha=0.7)
        
        ax.set_title('Attack Timeline - Findings per Phase', fontsize=16, color='white')
        ax.set_xlabel('Attack Phase', color='white')
        ax.set_ylabel('Vulnerabilities Found', color='white')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Convert to base64
        import io
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', facecolor='black', edgecolor='white', bbox_inches='tight')
        img_bytes.seek(0)
        
        img_base64 = base64.b64encode(img_bytes.read()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"
    
    def _get_html_template(self) -> str:
        """Get HTML report template"""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent DS - Penetration Test Report</title>
    <style>{{ css_styles }}</style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>AGENT DS - PENETRATION TEST REPORT</h1>
            <div class="report-info">
                <p><strong>Report ID:</strong> {{ metadata.report_id }}</p>
                <p><strong>Generated:</strong> {{ metadata.timestamp }}</p>
                <p><strong>Target:</strong> {{ metadata.target }}</p>
                <p><strong>Risk Level:</strong> <span class="risk-{{ executive_summary.overall_risk_level.lower() }}">{{ executive_summary.overall_risk_level }}</span></p>
            </div>
        </header>

        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <h3>{{ executive_summary.total_vulnerabilities }}</h3>
                    <p>Total Vulnerabilities</p>
                </div>
                <div class="stat-box critical">
                    <h3>{{ executive_summary.critical_vulnerabilities }}</h3>
                    <p>Critical Issues</p>
                </div>
                <div class="stat-box high">
                    <h3>{{ executive_summary.high_vulnerabilities }}</h3>
                    <p>High Risk</p>
                </div>
            </div>
            
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                {% for finding in executive_summary.key_points %}
                    <li>{{ finding }}</li>
                {% endfor %}
                </ul>
            </div>
        </section>

        {% if charts.severity_pie %}
        <section class="charts">
            <h2>Vulnerability Analysis</h2>
            <div class="chart-container">
                <img src="{{ charts.severity_pie }}" alt="Severity Distribution" class="chart">
            </div>
        </section>
        {% endif %}

        <section class="vulnerabilities">
            <h2>Vulnerability Details</h2>
            {% if vulnerability_details %}
            <div class="vuln-table">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Type</th>
                            <th>Severity</th>
                            <th>Description</th>
                            <th>Location</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for vuln in vulnerability_details %}
                        <tr class="severity-{{ vuln.severity.lower() }}">
                            <td>{{ vuln.id }}</td>
                            <td>{{ vuln.type }}</td>
                            <td><span class="severity-badge severity-{{ vuln.severity.lower() }}">{{ vuln.severity }}</span></td>
                            <td>{{ vuln.description }}</td>
                            <td class="location">{{ vuln.location }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            {% else %}
            <p class="no-vulns">No vulnerabilities detected.</p>
            {% endif %}
        </section>

        <section class="recommendations">
            <h2>Recommendations</h2>
            <div class="rec-section">
                <h3>Immediate Actions</h3>
                <ol>
                {% for action in recommendations.immediate_actions %}
                    <li>{{ action }}</li>
                {% endfor %}
                </ol>
            </div>
            
            <div class="rec-section">
                <h3>Long-term Improvements</h3>
                <ol>
                {% for improvement in recommendations.long_term_improvements %}
                    <li>{{ improvement }}</li>
                {% endfor %}
                </ol>
            </div>
        </section>

        <footer class="footer">
            <p>Generated by Agent DS v{{ metadata.agent_version }} | {{ metadata.timestamp }}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report"""
        
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #e0e0e0;
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #00ff41 0%, #00b33c 100%);
            border-radius: 10px;
            color: #000;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .report-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .report-info p {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 5px;
        }

        .risk-critical { color: #ff0000; font-weight: bold; }
        .risk-high { color: #ff8000; font-weight: bold; }
        .risk-medium { color: #ffff00; font-weight: bold; }
        .risk-low { color: #00ff00; font-weight: bold; }
        .risk-minimal { color: #00ffff; font-weight: bold; }

        section {
            margin-bottom: 40px;
            background: rgba(255,255,255,0.05);
            padding: 30px;
            border-radius: 10px;
            border: 1px solid rgba(0,255,65,0.2);
        }

        h2 {
            color: #00ff41;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 2px solid #00ff41;
            padding-bottom: 10px;
        }

        h3 {
            color: #00b33c;
            margin-bottom: 15px;
        }

        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-box {
            text-align: center;
            padding: 20px;
            background: rgba(0,255,65,0.1);
            border-radius: 8px;
            border: 1px solid #00ff41;
        }

        .stat-box.critical {
            background: rgba(255,0,0,0.1);
            border-color: #ff0000;
        }

        .stat-box.high {
            background: rgba(255,128,0,0.1);
            border-color: #ff8000;
        }

        .stat-box h3 {
            font-size: 2em;
            margin-bottom: 5px;
        }

        .key-findings ul {
            list-style-type: none;
        }

        .key-findings li {
            padding: 10px 0;
            border-bottom: 1px solid rgba(0,255,65,0.2);
        }

        .key-findings li::before {
            content: "→ ";
            color: #00ff41;
            font-weight: bold;
        }

        .chart-container {
            text-align: center;
            margin: 20px 0;
        }

        .chart {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }

        .vuln-table {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(0,255,65,0.2);
        }

        th {
            background: rgba(0,255,65,0.2);
            color: #00ff41;
            font-weight: bold;
        }

        .severity-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }

        .severity-critical {
            background: #ff0000;
            color: white;
        }

        .severity-high {
            background: #ff8000;
            color: white;
        }

        .severity-medium {
            background: #ffff00;
            color: black;
        }

        .severity-low {
            background: #00ff00;
            color: black;
        }

        .location {
            font-family: monospace;
            font-size: 0.9em;
            word-break: break-all;
        }

        .no-vulns {
            text-align: center;
            font-size: 1.2em;
            color: #00ff41;
            padding: 40px;
        }

        .rec-section {
            margin-bottom: 30px;
        }

        .rec-section ol {
            padding-left: 20px;
        }

        .rec-section li {
            margin-bottom: 10px;
            padding: 8px;
            background: rgba(0,255,65,0.05);
            border-left: 3px solid #00ff41;
        }

        .footer {
            text-align: center;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
            color: #888;
            margin-top: 40px;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            .summary-stats {
                grid-template-columns: 1fr;
            }
            
            th, td {
                padding: 8px 4px;
                font-size: 0.9em;
            }
        }
        """
    
    def _generate_simple_html(self, report_data: Dict[str, Any]) -> str:
        """Generate simple HTML without Jinja2"""
        
        # This is a simplified version for when Jinja2 is not available
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Agent DS Report</title>
    <style>{self.css_styles}</style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>AGENT DS - PENETRATION TEST REPORT</h1>
            <p><strong>Target:</strong> {report_data['metadata']['target']}</p>
            <p><strong>Generated:</strong> {report_data['metadata']['timestamp']}</p>
        </header>
        
        <section>
            <h2>Executive Summary</h2>
            <p><strong>Total Vulnerabilities:</strong> {report_data['executive_summary']['total_vulnerabilities']}</p>
            <p><strong>Risk Level:</strong> {report_data['executive_summary']['overall_risk_level']}</p>
        </section>
        
        <section>
            <h2>Vulnerabilities</h2>
"""
        
        for vuln in report_data['vulnerability_details'][:10]:
            html += f"""
            <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #00ff41; border-radius: 5px;">
                <h3>{vuln['type']} - {vuln['severity']}</h3>
                <p>{vuln['description']}</p>
                <p><strong>Location:</strong> {vuln['location']}</p>
            </div>
"""
        
        html += """
        </section>
        
        <footer class="footer">
            <p>Generated by Agent DS v2.0</p>
        </footer>
    </div>
</body>
</html>
"""
        return html


# Test function
async def main():
    """Test the reporting system"""
    
    # Mock attack results
    mock_results = {
        'target_url': 'https://example.com',
        'attack_type': 'One-Click Attack',
        'start_time': datetime.now() - timedelta(hours=1),
        'end_time': datetime.now(),
        'web_attacks': {
            'sql_injection': {
                'vulnerable': True,
                'vulnerabilities': [
                    {
                        'type': 'SQL Injection',
                        'severity': 'CRITICAL',
                        'confidence': 'high',
                        'description': 'Union-based SQL injection in login form',
                        'url': 'https://example.com/login.php',
                        'payload': "' UNION SELECT version()--",
                        'impact': 'Database access and data extraction possible'
                    }
                ]
            },
            'xss': {
                'vulnerable': True,
                'vulnerabilities': [
                    {
                        'type': 'Cross-Site Scripting',
                        'severity': 'HIGH',
                        'confidence': 'medium',
                        'description': 'Reflected XSS in search parameter',
                        'url': 'https://example.com/search?q=<script>alert(1)</script>',
                        'payload': '<script>alert(document.cookie)</script>',
                        'impact': 'Session hijacking and phishing attacks possible'
                    }
                ]
            }
        },
        'admin_login': {
            'credentials': [
                {'username': 'admin', 'password': 'admin', 'status': 'success'}
            ]
        }
    }
    
    generator = ReportGenerator()
    
    print("Generating reports...")
    
    # Generate JSON report
    await generator.generate_json_report(mock_results, Path("test_report.json"))
    print("✓ JSON report generated")
    
    # Generate HTML report
    await generator.generate_html_report(mock_results, Path("test_report.html"))
    print("✓ HTML report generated")
    
    # Generate text report
    await generator.generate_text_report(mock_results, Path("test_report.txt"))
    print("✓ Text report generated")
    
    # Generate PDF report if available
    if PDF_AVAILABLE:
        await generator.generate_pdf_report(mock_results, Path("test_report.pdf"))
        print("✓ PDF report generated")
    else:
        print("⚠ PDF generation not available (missing reportlab)")

if __name__ == "__main__":
    asyncio.run(main())