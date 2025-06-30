# AI Agent Test Generation Instructions

## 🎯 Objective
Generate comprehensive test coverage for JARVIS ecosystem components to achieve 80%+ coverage within 3-4 days.

## 🚀 Quick Start

```bash
# 1. Analyze coverage gaps
./tests/ai_test_generator.sh analyze

# 2. Generate tests for top 5 modules
./tests/ai_test_generator.sh batch

# 3. Or use Python batch processor
python3 tests/batch_test_generator.py --max 10 --parallel
```

## 📋 Process Overview

### Step 1: Identify Low Coverage Modules
```python
# The system automatically identifies modules below 80% coverage
# Results saved to: tests/metadata/modules_to_test.json
```

### Step 2: Analyze Module Structure
For each module, the system extracts:
- Classes and their methods (sync/async)
- Standalone functions
- External dependencies
- Complexity metrics

### Step 3: Generate Test Skeleton
Using `test_generator_template.py`, creates:
- Test class for each module class
- Test methods for all public methods
- Async test handling
- Property-based test stubs
- Mock placeholders

### Step 4: Fill in Test Implementation

## 🔧 Detailed Instructions for AI Agents

### 1. Reading Module Code
```python
# Always read the actual module code first
with open('core/module_name.py') as f:
    code = f.read()
```

### 2. Identify Test Requirements
- **Public API**: Test all public classes, methods, and functions
- **Async Methods**: Use `@pytest.mark.asyncio` decorator
- **Dependencies**: Mock all external services
- **Edge Cases**: Empty inputs, None values, extreme values
- **Error Cases**: Invalid inputs, missing resources, timeouts

### 3. Test Patterns to Follow

#### Basic Method Test
```python
def test_method_name(self, component):
    """Test method_name with various scenarios"""
    # Arrange - setup test data
    input_data = {"key": "value"}
    expected = {"result": "expected"}
    
    # Act - call the method
    result = component.method_name(input_data)
    
    # Assert - verify results
    assert result == expected
    assert component.state_updated == True
```

#### Async Method Test
```python
@pytest.mark.asyncio
async def test_async_method(self, component):
    """Test async operations"""
    # Mock async dependencies
    with patch('module.external_api.fetch') as mock_fetch:
        mock_fetch.return_value = {"data": "mocked"}
        
        result = await component.async_method()
        
        assert result is not None
        mock_fetch.assert_called_once()
```

#### Error Handling Test
```python
def test_error_handling(self, component):
    """Test error scenarios"""
    # Test invalid input
    with pytest.raises(ValueError) as exc_info:
        component.process(None)
    
    assert "Input cannot be None" in str(exc_info.value)
    
    # Test recovery
    with patch.object(component, 'external_service', side_effect=ConnectionError):
        result = component.process_with_fallback({"data": "test"})
        assert result["fallback_used"] == True
```

#### Property-Based Test
```python
@given(
    size=st.integers(min_value=0, max_value=1000),
    data_type=st.sampled_from(['list', 'dict', 'string'])
)
def test_handles_any_size(self, component, size, data_type):
    """Test with various input sizes and types"""
    if data_type == 'list':
        input_data = ['item'] * size
    elif data_type == 'dict':
        input_data = {f'key_{i}': i for i in range(size)}
    else:
        input_data = 'x' * size
    
    # Should handle any valid input
    result = component.process(input_data)
    assert result is not None
```

### 4. Mocking External Dependencies

#### Using Existing Mocks
```python
# From conftest.py
def test_with_redis(self, component, mock_redis):
    """Test Redis integration"""
    mock_redis.get.return_value = "cached_value"
    
    result = component.get_from_cache("key")
    
    assert result == "cached_value"
    mock_redis.get.assert_called_with("key")
```

#### Creating Custom Mocks
```python
# For module-specific dependencies
@patch('core.module.SpecialService')
def test_with_special_service(self, mock_service, component):
    """Test with mocked special service"""
    mock_instance = Mock()
    mock_instance.process.return_value = {"status": "ok"}
    mock_service.return_value = mock_instance
    
    result = component.use_special_service()
    
    assert result["status"] == "ok"
```

### 5. Test Organization

#### Group Related Tests
```python
class TestComponentCore:
    """Core functionality tests"""
    
    def test_initialization(self): ...
    def test_basic_operation(self): ...

class TestComponentEdgeCases:
    """Edge case tests"""
    
    def test_empty_input(self): ...
    def test_large_input(self): ...

class TestComponentIntegration:
    """Integration tests"""
    
    @pytest.mark.integration
    def test_with_database(self): ...
```

### 6. Coverage Focus Areas

#### High Priority (Test First)
1. **Core Business Logic**: Main functionality
2. **Error Handling**: Exception paths
3. **Async Operations**: Concurrent code
4. **State Management**: Stateful operations
5. **Integration Points**: External service calls

#### Lower Priority
1. **Simple Getters/Setters**: Unless complex logic
2. **Logging Statements**: Usually excluded
3. **Main Blocks**: `if __name__ == "__main__"`

### 7. Validation Checklist

Before finalizing a test file:
- [ ] All public methods have tests
- [ ] Async methods use proper decorators
- [ ] External dependencies are mocked
- [ ] Error cases are covered
- [ ] Tests are independent (no shared state)
- [ ] Tests run quickly (mock slow operations)
- [ ] Coverage > 80% for the module

### 8. Common Pitfalls to Avoid

1. **Don't Test Implementation Details**: Test behavior, not internals
2. **Avoid Brittle Tests**: Don't assert on exact strings unless critical
3. **Mock External Services**: Never hit real APIs/databases
4. **Keep Tests Fast**: Mock sleep(), time-consuming operations
5. **Test One Thing**: Each test should verify one behavior

## 📊 Progress Tracking

### Check Current Coverage
```bash
# See overall coverage
pytest --cov --cov-report=term

# See specific module coverage
pytest tests/test_module_name.py --cov=core.module_name --cov-report=term-missing
```

### Daily Progress
```bash
# Run daily coverage check
./run_daily_coverage.sh

# Compare with baseline
diff coverage_reports/baseline_coverage.txt coverage_reports/daily_*.txt
```

## 🎯 Success Metrics

### Day 1 Goals
- Generate tests for 10-15 modules
- Achieve 40% overall coverage
- Focus on core components

### Day 2 Goals
- Generate tests for 15-20 modules
- Achieve 60% overall coverage
- Add integration tests

### Day 3 Goals
- Generate tests for remaining modules
- Achieve 75% overall coverage
- Add edge case tests

### Day 4 Goals
- Polish and optimize tests
- Achieve 80%+ overall coverage
- Add performance tests

## 🛠️ Tools and Commands

### Generate Tests
```bash
# Single module
./tests/ai_test_generator.sh generate core/module_name.py

# Batch processing
python3 tests/batch_test_generator.py --max 20

# With parallelization
python3 tests/batch_test_generator.py --max 20 --parallel
```

### Validate Tests
```bash
# Validate single test
./tests/ai_test_generator.sh validate tests/test_module_generated.py

# Run all generated tests
pytest tests/*_generated.py -v
```

### Finalize Tests
```bash
# Move generated tests to final location
python3 tests/batch_test_generator.py --finalize

# Generate coverage report
python3 tests/batch_test_generator.py --report
```

## 📝 Example Transformation

### From Module:
```python
# core/data_processor.py
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    async def process_data(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Process logic
        result = await self._transform(data)
        self.cache[data['id']] = result
        return result
    
    async def _transform(self, data):
        # Complex transformation
        return {"transformed": data}
```

### To Test:
```python
# tests/test_data_processor.py
class TestDataProcessor:
    @pytest.fixture
    async def processor(self):
        config = {"timeout": 30}
        processor = DataProcessor(config)
        yield processor
        # Cleanup
        processor.cache.clear()
    
    @pytest.mark.asyncio
    async def test_process_data_success(self, processor):
        """Test successful data processing"""
        data = {"id": "123", "value": "test"}
        
        result = await processor.process_data(data)
        
        assert result == {"transformed": data}
        assert processor.cache["123"] == result
    
    @pytest.mark.asyncio
    async def test_process_data_empty_error(self, processor):
        """Test error on empty data"""
        with pytest.raises(ValueError) as exc_info:
            await processor.process_data(None)
        
        assert "Data cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_transform_mocked(self, processor):
        """Test with mocked transform"""
        with patch.object(processor, '_transform') as mock_transform:
            mock_transform.return_value = {"mocked": True}
            
            result = await processor.process_data({"id": "456"})
            
            assert result == {"mocked": True}
            mock_transform.assert_called_once()
```

## 🚀 Get Started Now!

1. Run coverage analysis: `./tests/ai_test_generator.sh analyze`
2. Generate first batch: `./tests/ai_test_generator.sh batch`
3. Review generated tests in `tests/generated/`
4. Fill in TODO sections with actual test logic
5. Run tests and iterate

Remember: The goal is 80%+ coverage with meaningful tests that ensure JARVIS reliability!