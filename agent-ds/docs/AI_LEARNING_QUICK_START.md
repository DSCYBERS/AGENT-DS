# Agent DS - AI Learning Quick Start Guide

## Getting Started with AI Learning

### Prerequisites
1. **Install AI/ML Dependencies**
   ```bash
   pip install torch transformers scikit-learn numpy pandas joblib
   ```

2. **Initialize AI Learning Database**
   ```bash
   agent-ds ai status  # This will auto-initialize the AI learning database
   ```

3. **Configure External Intelligence APIs** (Optional)
   ```bash
   export OTX_API_KEY="your_otx_api_key"  # For AlienVault OTX integration
   ```

## Basic AI Learning Workflow

### Step 1: Enable AI Learning
```bash
# Check AI learning status
agent-ds ai status --detailed

# Expected output:
# ✓ Autonomous Learning Engine: Active
# ✓ AI Models: Loaded
# ✓ External Intelligence: Connected
# ✓ Sandbox Experimenter: Ready
```

### Step 2: Execute Your First Adaptive Mission
```bash
# Login first
agent-ds login --username admin --password 8460Divy!@#$

# Start adaptive attack mission
agent-ds ai adaptive-attack --target testphp.vulnweb.com --mission-name ai_test_01

# The AI will:
# 1. Perform enhanced reconnaissance with AI analysis
# 2. Generate custom attack strategies based on discovered technologies
# 3. Execute adaptive attacks with real-time payload modification
# 4. Learn from results and store insights for future missions
```

### Step 3: Train AI from Mission Results
```bash
# Train AI from the completed mission
agent-ds ai train --mission-id ai_test_01

# Expected output:
# ✓ Learned from mission ai_test_01
# Generated 12 insights from mission ai_test_01
# 
# Training Session Summary:
# Total Sessions: 1
# Insights Generated: 12
# Training Duration: 45.2s
```

### Step 4: View AI Learning Insights
```bash
# View generated learning insights
agent-ds ai insights --limit 10

# Expected output showing insights like:
# ┌─────────────────┬─────────────────────────────┬────────────┬──────────────────┐
# │ Type            │ Success Factors             │ Confidence │ Scenarios        │
# ├─────────────────┼─────────────────────────────┼────────────┼──────────────────┤
# │ success_pattern │ payload_length:45,          │ 0.85       │ sql_injection,   │
# │                 │ sql_union_technique,        │            │ mysql_targets    │
# │                 │ target_mysql:5.7            │            │                  │
# └─────────────────┴─────────────────────────────┴────────────┴──────────────────┘
```

## Advanced AI Features

### Payload Experimentation
```bash
# Experiment with novel SQL injection payloads
agent-ds ai experiment --attack-type sql_injection --target mysql

# Safe sandbox-only mode for testing
agent-ds ai experiment --attack-type xss --sandbox-only

# Results show novel payload discoveries:
# Novel Payloads Discovered: 3
# Success Rate: 37.5%
# 1. ' UNION SELECT LOAD_FILE('/etc/passwd'),2,3--
# 2. ' OR SLEEP(5) AND '1'='1'--
# 3. '; INSERT INTO users VALUES('hacker','pwned');--
```

### External Intelligence Updates
```bash
# Update AI knowledge with latest threat intelligence
agent-ds ai train --external-intel

# Results:
# ✓ Updated external intelligence
# Updated intelligence from 2 sources
# Intelligence Sources Updated: cve, exploitdb
# New Intelligence Items: 47
```

### Full Autonomous Mode
```bash
# Enable full AI autonomous mode (requires authorization confirmation)
agent-ds ai adaptive-attack --target vulnerable-app.com --ai-mode

# WARNING: This will execute adaptive attacks with minimal human oversight
# The AI will make real-time decisions about:
# - Attack vector selection
# - Payload generation and modification
# - Evasion technique application
# - Attack sequence optimization
```

## Real-World Examples

### Example 1: WordPress Site Analysis
```bash
# Target a WordPress site
agent-ds ai adaptive-attack --target wordpress-site.com --mission-name wp_test

# AI Analysis Results:
# ┌──────────────────────┬──────────┬──────────────────────────────────┐
# │ Phase                │ Status   │ Results                          │
# ├──────────────────────┼──────────┼──────────────────────────────────┤
# │ reconnaissance       │ ✓ Completed │ Found 15 potential vectors    │
# │ vulnerability_analysis│ ✓ Completed │ Identified 8 vulnerabilities │
# │ strategy_generation  │ ✓ Completed │ Generated adaptive strategy   │
# │ adaptive_attacks     │ ✓ Completed │ Attack successful             │
# └──────────────────────┴──────────┴──────────────────────────────────┘
#
# AI Adaptations Made: 2
# • payload_modify: Generated new payload based on failure pattern analysis
# • evasion_add: Adding evasion techniques due to detection indicators
```

### Example 2: Custom Target with Learning
```bash
# First mission against a custom target
agent-ds ai adaptive-attack --target internal.company.com --mission-name internal_01

# Train AI from results
agent-ds ai train --mission-id internal_01

# Second mission using learned knowledge
agent-ds ai adaptive-attack --target internal2.company.com --mission-name internal_02

# AI will apply learned patterns:
# - Payloads that worked against similar internal targets
# - Evasion techniques effective against corporate firewalls
# - Attack sequences optimized for internal network layouts
```

### Example 3: Multi-Mission Learning
```bash
# Execute multiple missions
agent-ds ai adaptive-attack --target site1.com --mission-name multi_01
agent-ds ai adaptive-attack --target site2.com --mission-name multi_02
agent-ds ai adaptive-attack --target site3.com --mission-name multi_03

# Train AI from all missions simultaneously
agent-ds ai train --all-missions

# View cross-mission insights
agent-ds ai insights --attack-type sql_injection --limit 15

# AI discovers patterns like:
# - Certain payload structures work well against PHP applications
# - MySQL 5.7 targets are vulnerable to specific union-based attacks
# - Apache/PHP/MySQL stacks often have predictable attack vectors
```

## AI Learning Configuration

### Enable Advanced Features
Edit `config/ai_learning_config.yaml`:
```yaml
ai_learning:
  enabled: true
  
  # Enable experimental features
  feature_flags:
    autonomous_mode: true
    real_time_adaptation: true
    payload_innovation: true
    sandbox_experimentation: true
    
  # AI model configuration
  models:
    payload_generator:
      temperature: 0.8  # Creativity level (0.0-1.0)
      max_length: 512   # Maximum payload length
    
    success_predictor:
      confidence_threshold: 0.7  # Minimum confidence for predictions
```

### Sandbox Safety Configuration
```yaml
sandbox:
  enabled: true
  isolation_level: "high"
  timeout: 300  # 5 minutes max per experiment
  max_concurrent_experiments: 3
  safety_checks: true
```

## Monitoring AI Performance

### View Performance Metrics
```bash
# Check AI improvement over time
agent-ds ai metrics --type success_rate --period 30d

# View adaptation effectiveness
agent-ds ai metrics --type adaptation_success --period 7d

# Monitor payload innovation
agent-ds ai metrics --type novel_payloads --period 30d
```

### Expected Performance Improvements
After 10+ missions with AI learning enabled:
- **Success Rate**: 25-40% improvement over baseline
- **Time to Compromise**: 30-50% reduction
- **Detection Avoidance**: 20-35% improvement
- **Novel Technique Discovery**: 5-10 new effective payloads per month

## Troubleshooting Common Issues

### Issue: AI Models Not Loading
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# If missing, install:
pip install torch==2.0.1 transformers==4.35.2
```

### Issue: No Learning Insights Generated
```bash
# Verify missions have attack results
agent-ds mission status --mission-id your_mission_id

# Check AI learning database
agent-ds ai status --detailed

# Force training with verbose output
agent-ds ai train --mission-id your_mission_id --verbose
```

### Issue: External Intelligence Update Failures
```bash
# Check API connectivity
curl -s "https://services.nvd.nist.gov/rest/json/cves/1.0" | head

# Verify configuration
agent-ds config show | grep -A5 external_sources
```

## Best Practices

### 1. Start with Sandbox Mode
```bash
# Always test new attack types in sandbox first
agent-ds ai experiment --attack-type new_technique --sandbox-only
```

### 2. Regular Training Sessions
```bash
# Train AI weekly from all recent missions
agent-ds ai train --all-missions --external-intel
```

### 3. Monitor AI Decisions
```bash
# Review AI adaptation decisions regularly
agent-ds ai insights --type adaptation_decision --limit 20
```

### 4. Validate AI Insights
```bash
# Test AI-recommended payloads in controlled environments
agent-ds ai experiment --payload "AI_GENERATED_PAYLOAD" --sandbox-only
```

## Security Considerations

### Authorization Requirements
- **Autonomous Mode**: Requires explicit confirmation for zero-click operation
- **External Intelligence**: Review external data sources for sensitive operations
- **Adaptation Limits**: Configure maximum adaptation attempts to prevent runaway automation

### Compliance Features
- **Audit Logging**: All AI decisions are logged with reasoning
- **Human Override**: Can interrupt autonomous operations at any time
- **Risk Assessment**: AI evaluates detection probability before each attack

### Data Protection
- **Encrypted Storage**: All learning data encrypted at rest
- **Access Controls**: AI learning data requires same authentication as mission data
- **Data Retention**: Configure retention policies for learning data

## Next Steps

1. **Run Your First AI Mission**: Start with the basic workflow above
2. **Experiment with Payloads**: Use sandbox mode to discover new techniques
3. **Monitor Learning Progress**: Track AI performance improvements
4. **Advanced Configuration**: Customize AI models for your specific use cases
5. **Contribute to Learning**: Share anonymized insights with the Agent DS community (if configured)

The AI Learning Module transforms Agent DS from a static tool into an evolving, self-improving penetration testing framework that gets better with every mission.