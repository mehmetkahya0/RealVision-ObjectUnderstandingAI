# 🔬 Tests Directory

This directory contains all test files for validating the application functionality.

## 🧪 Test Files

### Import & Environment Tests
- `test_imports.py` - Verifies all required libraries are installed and working
- Tests Python environment, library versions, and basic functionality

### Data Science Tests
- `test_data_science.py` - Tests analytics and data science features
- Validates performance analysis, statistical computations, and data export

### Model Analysis Tests
- `test_models_analyze.py` - Tests model analysis functionality
- Validates model comparison, performance metrics, and report generation

### Visualization Tests
- `test_visualization_system.py` - Tests the complete visualization system
- Validates data loading, chart generation, and GUI functionality

## 🏃 Running Tests

### Individual Tests
```bash
# Test imports and environment
python tests/test_imports.py

# Test data science features  
python tests/test_data_science.py

# Test visualization system
python tests/test_visualization_system.py

# Test model analysis
python tests/test_models_analyze.py
```

### All Tests
```bash
# Run comprehensive test suite
python app.py test
```

## ✅ Test Coverage

The test suite covers:
- ✅ Library imports and versions
- ✅ Camera detection and access
- ✅ Model loading and inference
- ✅ Data science and analytics functions
- ✅ Visualization and reporting tools
- ✅ File I/O and data persistence
- ✅ GUI components and interactions

## 🐛 Debugging Tests

If tests fail:
1. Check the error messages for specific issues
2. Verify virtual environment is activated
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Run tests individually to isolate problems

*Note: Tests are designed to be run from the project root directory.*
