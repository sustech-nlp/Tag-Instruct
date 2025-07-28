# Baseline Comparison Methods

This directory contains implementations of four baseline instruction complexity enhancement methods for comparison with Tag-Instruct.

## 📁 Implemented Methods

### 1. **WizardLM** (`wizardlm.py`)
- Official implementation following the original WizardLM paper
- Evolution-based instruction augmentation using four operations: constraints, deepening, concretizing, and reasoning

### 2. **Tree-Instruct** (`tree-instruct.py`) 
- Implementation based on the Tree-Instruct methodology
- Semantic tree expansion approach for instruction complexity enhancement

### 3. **CodeCLM** (`codeclm-.py`)
- Reproduced implementation based on the CodeCLM paper (no official code available)
- Domain-specific rubrics and action-based instruction improvement
- Removed instruction cleaning operations to focus on complexity augmentation

### 4. **AutoEvol** (`autoevol-general.py`)
- Reproduced implementation based on the AutoEvol paper (no official code available)
- Automatic evolution of instructions through iterative rewriting

## 🎯 Implementation Notes

- **Alignment with Official Code**: Methods with available official implementations (WizardLM, Tree-Instruct) follow the original codebase as closely as possible
- **Paper-based Reproduction**: For methods without open-sourced code (CodeCLM, AutoEvol), we implemented based on the methodologies described in their respective papers
- **Focus on Complexity**: All implementations prioritize **instruction difficulty enhancement** over other aspects like data filtering or diversity augmentation
- **Simplified Pipeline**: Data filtering and diversity enhancement components have been commented out to provide a fair comparison of complexity augmentation capabilities

## 🔧 Usage

Each method can be run independently with similar configuration patterns. Update the input/output paths and model configurations as needed for your experimental setup.

## 📊 Comparison Focus

These implementations enable direct comparison of different instruction complexity enhancement approaches while maintaining consistent experimental conditions focused on difficulty augmentation.