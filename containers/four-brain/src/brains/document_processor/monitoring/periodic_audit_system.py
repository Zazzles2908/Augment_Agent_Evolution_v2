"""
Periodic Anti-Fabrication Audit System - ZERO FABRICATION POLICY
Automated scanning for fabrication patterns in codebase
NO FABRICATION - Real code analysis with honest findings
"""

import os
import re
import ast
import logging
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

from ..config.settings import Brain4Settings

logger = logging.getLogger(__name__)

@dataclass
class FabricationPattern:
    """Fabrication pattern detection result"""
    file_path: str
    line_number: int
    pattern_type: str
    severity: str
    code_snippet: str
    description: str
    recommendation: str

@dataclass
class AuditReport:
    """Periodic audit report"""
    audit_id: str
    timestamp: str
    files_scanned: int
    patterns_detected: int
    high_severity_issues: int
    medium_severity_issues: int
    low_severity_issues: int
    fabrication_patterns: List[FabricationPattern]
    compliance_score: float
    recommendations: List[str]

class PeriodicAuditSystem:
    """
    Periodic Anti-Fabrication Audit System - ZERO FABRICATION POLICY
    Automated detection of fabrication patterns in codebase
    """
    
    def __init__(self, settings: Brain4Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Audit configuration
        self.source_directory = Path("src/brain4_docling")
        self.audit_history: List[AuditReport] = []
        
        # Fabrication patterns to detect
        self.fabrication_patterns = {
            # Hardcoded success responses
            "hardcoded_success": {
                "patterns": [
                    r'return\s*{\s*["\']status["\']\s*:\s*["\']success["\']',
                    r'return\s*{\s*["\']success["\']\s*:\s*True',
                    r'"status":\s*"success".*"message":\s*".*successfully.*"',
                    r'return\s*True\s*#.*success'
                ],
                "severity": "high",
                "description": "Hardcoded success response without actual processing"
            },
            
            # Mock/fake implementations
            "mock_implementations": {
                "patterns": [
                    r'def\s+\w+.*:\s*#.*mock|fake|placeholder',
                    r'return\s*\[\].*#.*mock|fake|empty',
                    r'mock\w*\s*=.*return_value',
                    r'@patch|@mock',
                    r'Mock\(|MagicMock\(|AsyncMock\('
                ],
                "severity": "medium",
                "description": "Mock or fake implementation detected"
            },
            
            # Fabricated data generation
            "fabricated_data": {
                "patterns": [
                    r'fake_\w+\s*=',
                    r'dummy_\w+\s*=',
                    r'test_\w+\s*=.*\[.*\].*#.*fake',
                    r'return\s*\[0\.1,\s*0\.2.*\].*#.*fake|dummy|test'
                ],
                "severity": "high",
                "description": "Fabricated data generation detected"
            },
            
            # Fake processing delays
            "fake_delays": {
                "patterns": [
                    r'time\.sleep\(.*\).*#.*simulate|fake',
                    r'await\s+asyncio\.sleep\(.*\).*#.*simulate|fake'
                ],
                "severity": "medium",
                "description": "Fake processing delay detected"
            },
            
            # Exception swallowing with fake success
            "exception_swallowing": {
                "patterns": [
                    r'except.*:\s*return\s*True',
                    r'except.*:\s*pass.*return.*success',
                    r'except.*Exception.*:\s*return\s*{\s*["\']success["\']'
                ],
                "severity": "high",
                "description": "Exception swallowing with fabricated success"
            },
            
            # Hardcoded embeddings/vectors
            "hardcoded_vectors": {
                "patterns": [
                    r'return\s*\[\[0\.1.*0\.2.*0\.3.*\]\]',
                    r'embeddings\s*=\s*\[\[.*\]\].*#.*fake|dummy',
                    r'np\.zeros\(.*\).*#.*fake|placeholder'
                ],
                "severity": "high",
                "description": "Hardcoded embedding vectors detected"
            },
            
            # Fake status reporting
            "fake_status": {
                "patterns": [
                    r'["\']status["\']\s*:\s*["\']completed["\'].*#.*fake',
                    r'["\']processing_status["\']\s*:\s*["\']queued["\'].*#.*fake',
                    r'task_status\s*=\s*["\']success["\'].*#.*fake'
                ],
                "severity": "medium",
                "description": "Fake status reporting detected"
            }
        }
    
    async def conduct_audit(self) -> AuditReport:
        """Conduct comprehensive anti-fabrication audit"""
        audit_start = time.time()
        audit_id = f"audit_{int(audit_start)}"
        
        self.logger.info(f"🔍 Starting periodic anti-fabrication audit: {audit_id}")
        
        try:
            # Scan all Python files
            python_files = self._get_python_files()
            fabrication_patterns = []
            
            for file_path in python_files:
                patterns = await self._scan_file_for_fabrication(file_path)
                fabrication_patterns.extend(patterns)
            
            # Analyze results
            high_severity = len([p for p in fabrication_patterns if p.severity == "high"])
            medium_severity = len([p for p in fabrication_patterns if p.severity == "medium"])
            low_severity = len([p for p in fabrication_patterns if p.severity == "low"])
            
            # Calculate compliance score
            total_patterns = len(fabrication_patterns)
            compliance_score = max(0, 100 - (high_severity * 20 + medium_severity * 10 + low_severity * 5))
            
            # Generate recommendations
            recommendations = self._generate_audit_recommendations(fabrication_patterns)
            
            # Create audit report
            audit_report = AuditReport(
                audit_id=audit_id,
                timestamp=datetime.now().isoformat(),
                files_scanned=len(python_files),
                patterns_detected=total_patterns,
                high_severity_issues=high_severity,
                medium_severity_issues=medium_severity,
                low_severity_issues=low_severity,
                fabrication_patterns=fabrication_patterns,
                compliance_score=compliance_score,
                recommendations=recommendations
            )
            
            # Store audit report
            self.audit_history.append(audit_report)
            
            # Log results
            audit_time = time.time() - audit_start
            self.logger.info(f"✅ Audit completed: {total_patterns} patterns detected in {len(python_files)} files ({audit_time:.2f}s)")
            
            if total_patterns > 0:
                self.logger.warning(f"⚠️ Fabrication patterns detected: {high_severity} high, {medium_severity} medium, {low_severity} low severity")
            else:
                self.logger.info("🎯 No fabrication patterns detected - system compliant")
            
            return audit_report
            
        except Exception as e:
            self.logger.error(f"Audit failed: {e}")
            # Return honest error report
            return AuditReport(
                audit_id=audit_id,
                timestamp=datetime.now().isoformat(),
                files_scanned=0,
                patterns_detected=999,  # Error indicator
                high_severity_issues=999,
                medium_severity_issues=0,
                low_severity_issues=0,
                fabrication_patterns=[],
                compliance_score=0.0,
                recommendations=[f"Audit system error: {str(e)}"]
            )
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files to scan"""
        python_files = []
        
        if self.source_directory.exists():
            for file_path in self.source_directory.rglob("*.py"):
                # Skip test files and __pycache__
                if "__pycache__" not in str(file_path) and not file_path.name.startswith("test_"):
                    python_files.append(file_path)
        
        return python_files
    
    async def _scan_file_for_fabrication(self, file_path: Path) -> List[FabricationPattern]:
        """Scan individual file for fabrication patterns"""
        patterns_found = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Skip comments and empty lines
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                
                # Check each fabrication pattern
                for pattern_name, pattern_config in self.fabrication_patterns.items():
                    for regex_pattern in pattern_config["patterns"]:
                        if re.search(regex_pattern, line, re.IGNORECASE):
                            pattern = FabricationPattern(
                                file_path=str(file_path.relative_to(Path.cwd())),
                                line_number=line_num,
                                pattern_type=pattern_name,
                                severity=pattern_config["severity"],
                                code_snippet=line_stripped,
                                description=pattern_config["description"],
                                recommendation=self._get_pattern_recommendation(pattern_name)
                            )
                            patterns_found.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
        
        return patterns_found
    
    def _get_pattern_recommendation(self, pattern_type: str) -> str:
        """Get recommendation for specific pattern type"""
        recommendations = {
            "hardcoded_success": "Replace with authentic processing logic and real success validation",
            "mock_implementations": "Replace mocks with real component integration or honest placeholders",
            "fabricated_data": "Use real data sources or honest empty responses",
            "fake_delays": "Remove fake delays or replace with real processing time",
            "exception_swallowing": "Implement honest error handling with real failure reporting",
            "hardcoded_vectors": "Use real embedding generation or honest failure responses",
            "fake_status": "Implement real status tracking with authentic state management"
        }
        
        return recommendations.get(pattern_type, "Review and replace with authentic implementation")
    
    def _generate_audit_recommendations(self, patterns: List[FabricationPattern]) -> List[str]:
        """Generate audit recommendations"""
        recommendations = []
        
        if not patterns:
            recommendations.append("✅ No fabrication patterns detected - maintain current standards")
            return recommendations
        
        # Count pattern types
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
        
        # Generate specific recommendations
        for pattern_type, count in pattern_counts.items():
            if count > 0:
                recommendations.append(f"Address {count} instances of {pattern_type.replace('_', ' ')}")
        
        # Priority recommendations
        high_severity_count = len([p for p in patterns if p.severity == "high"])
        if high_severity_count > 0:
            recommendations.insert(0, f"🚨 URGENT: Fix {high_severity_count} high-severity fabrication patterns immediately")
        
        # Files with most issues
        file_counts = {}
        for pattern in patterns:
            file_counts[pattern.file_path] = file_counts.get(pattern.file_path, 0) + 1
        
        if file_counts:
            worst_file = max(file_counts.items(), key=lambda x: x[1])
            recommendations.append(f"Focus on {worst_file[0]} ({worst_file[1]} patterns detected)")
        
        return recommendations
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit system summary"""
        if not self.audit_history:
            return {
                "total_audits": 0,
                "last_audit": None,
                "compliance_trend": "no_data",
                "current_compliance": 0.0
            }
        
        latest_audit = self.audit_history[-1]
        
        # Calculate compliance trend
        if len(self.audit_history) >= 2:
            previous_score = self.audit_history[-2].compliance_score
            current_score = latest_audit.compliance_score
            trend = "improving" if current_score > previous_score else "declining" if current_score < previous_score else "stable"
        else:
            trend = "initial"
        
        return {
            "total_audits": len(self.audit_history),
            "last_audit": asdict(latest_audit),
            "compliance_trend": trend,
            "current_compliance": latest_audit.compliance_score,
            "zero_fabrication_compliant": latest_audit.patterns_detected == 0,
            "audit_frequency": "periodic",
            "next_audit_recommended": datetime.now().isoformat()
        }
    
    def save_audit_report(self, audit_report: AuditReport, output_path: Optional[Path] = None) -> Path:
        """Save audit report to file"""
        if output_path is None:
            output_path = Path(f"docs/QA/periodic_audit_report_{audit_report.audit_id}.md")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown report
        report_content = self._generate_markdown_report(audit_report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Audit report saved: {output_path}")
        return output_path
    
    def _generate_markdown_report(self, audit_report: AuditReport) -> str:
        """Generate markdown audit report"""
        report = f"""# Periodic Anti-Fabrication Audit Report
**Audit ID**: {audit_report.audit_id}  
**Timestamp**: {audit_report.timestamp}  
**Zero Fabrication Policy**: ENFORCED

## 📊 AUDIT SUMMARY

- **Files Scanned**: {audit_report.files_scanned}
- **Patterns Detected**: {audit_report.patterns_detected}
- **Compliance Score**: {audit_report.compliance_score:.1f}/100
- **Zero Fabrication Compliant**: {'✅ YES' if audit_report.patterns_detected == 0 else '❌ NO'}

### Severity Breakdown
- **🚨 High Severity**: {audit_report.high_severity_issues} issues
- **⚠️ Medium Severity**: {audit_report.medium_severity_issues} issues  
- **ℹ️ Low Severity**: {audit_report.low_severity_issues} issues

## 🔍 FABRICATION PATTERNS DETECTED

"""
        
        if audit_report.fabrication_patterns:
            for pattern in audit_report.fabrication_patterns:
                severity_icon = "🚨" if pattern.severity == "high" else "⚠️" if pattern.severity == "medium" else "ℹ️"
                report += f"""### {severity_icon} {pattern.pattern_type.replace('_', ' ').title()}

**File**: `{pattern.file_path}`  
**Line**: {pattern.line_number}  
**Severity**: {pattern.severity.upper()}  
**Code**: `{pattern.code_snippet}`  
**Description**: {pattern.description}  
**Recommendation**: {pattern.recommendation}

"""
        else:
            report += "✅ **No fabrication patterns detected** - System is compliant with zero fabrication policy.\n\n"
        
        report += f"""## 📋 RECOMMENDATIONS

"""
        for rec in audit_report.recommendations:
            report += f"- {rec}\n"
        
        report += f"""
---
**Report Generated**: {audit_report.timestamp}  
**Audit System**: Periodic Anti-Fabrication Scanner  
**Zero Fabrication Policy**: ENFORCED - All patterns must be eliminated
"""
        
        return report

# Global audit system instance
periodic_audit_system = None

async def get_audit_system(settings: Brain4Settings) -> PeriodicAuditSystem:
    """Get global audit system instance"""
    global periodic_audit_system
    
    if periodic_audit_system is None:
        periodic_audit_system = PeriodicAuditSystem(settings)
    
    return periodic_audit_system
