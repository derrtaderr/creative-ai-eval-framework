# Contributing to Creative AI Evaluation Framework

Thank you for your interest in contributing to the Creative AI Evaluation Framework! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### 💡 Ways to Contribute

1. **🐛 Bug Reports**: Help us identify and fix issues
2. **✨ Feature Requests**: Suggest new functionality
3. **📝 Documentation**: Improve guides, tutorials, and examples
4. **💻 Code Contributions**: Implement new features or fix bugs
5. **🧪 Testing**: Add test cases and improve coverage
6. **📊 Data**: Contribute sample datasets and benchmarks
7. **🎨 Examples**: Create real-world implementation examples

### 🚀 Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/creative-ai-eval-framework.git
   cd creative-ai-eval-framework
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Run Tests**
   ```bash
   pytest tests/
   ```

## 🔧 Development Workflow

### Branch Naming Convention
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test additions/improvements

### Commit Message Format
```
type(scope): short description

Longer description if needed

Closes #issue-number
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

**Example**:
```
feat(evaluators): add multi-modal coherence scoring

Implements cross-modal semantic alignment assessment for text, 
image, and video content evaluation.

Closes #42
```

## 📋 Contribution Guidelines

### Code Standards

1. **Python Style**
   - Follow PEP 8 style guide
   - Use `black` for code formatting
   - Use `flake8` for linting
   - Add type hints using `mypy`

2. **Testing Requirements**
   - Write unit tests for all new functionality
   - Maintain >90% test coverage
   - Include integration tests for evaluators
   - Add performance benchmarks for new features

3. **Documentation Standards**
   - Document all public methods and classes
   - Include docstring examples
   - Update relevant markdown documentation
   - Add notebook examples for new features

### Pull Request Process

1. **Before Submitting**
   ```bash
   # Format code
   black src/ tests/
   
   # Check linting
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest tests/ --cov=src --cov-report=term-missing
   ```

2. **PR Requirements**
   - Clear title and description
   - Link to related issues
   - Include test coverage for new code
   - Update documentation if needed
   - Add changelog entry if significant

3. **Review Process**
   - All PRs require review from maintainers
   - Address feedback promptly
   - Keep PRs focused and reasonably sized
   - Squash commits before merging

## 🎓 Development Guidelines

### Adding New Evaluators

1. **File Structure**
   ```
   src/evaluators/
   ├── __init__.py
   ├── base_evaluator.py
   ├── your_new_evaluator.py
   └── utils/
   ```

2. **Evaluator Template**
   ```python
   from .base_evaluator import BaseEvaluator
   
   class YourNewEvaluator(BaseEvaluator):
       """Brief description of evaluator purpose."""
       
       def __init__(self, config: dict = None):
           super().__init__(config)
           # Initialize evaluator-specific components
       
       def evaluate(self, content: str, context: dict = None) -> dict:
           """Main evaluation method."""
           # Implementation here
           return {"score": 0.0, "details": {}}
   ```

3. **Required Components**
   - Inherit from `BaseEvaluator`
   - Implement `evaluate()` method
   - Add comprehensive docstrings
   - Include unit tests
   - Add notebook example

### Documentation Standards

1. **Code Documentation**
   ```python
   def evaluate_content(self, content: str, creator_profile: dict, 
                       platform: str = "twitter") -> dict:
       """Evaluate content against creator profile and platform requirements.
       
       Args:
           content: The text content to evaluate
           creator_profile: Creator's voice and preference data
           platform: Target social media platform
           
       Returns:
           Dictionary containing evaluation scores and details
           
       Example:
           >>> evaluator = ContentContextEvaluator()
           >>> score = evaluator.evaluate_content(
           ...     "Great insights on AI trends!",
           ...     creator_profile,
           ...     platform="linkedin"
           ... )
           >>> print(score['context_score'])
           0.87
       """
   ```

2. **Markdown Documentation**
   - Use clear headings and structure
   - Include code examples
   - Add screenshots for UI components
   - Link to related documentation

### Testing Guidelines

1. **Unit Tests**
   ```python
   import pytest
   from src.evaluators import ContentContextEvaluator
   
   class TestContentContextEvaluator:
       def setup_method(self):
           self.evaluator = ContentContextEvaluator()
           self.sample_profile = {...}
       
       def test_evaluate_content_basic(self):
           result = self.evaluator.evaluate_content(
               "Test content", self.sample_profile
           )
           assert "context_score" in result
           assert 0 <= result["context_score"] <= 1
   ```

2. **Integration Tests**
   - Test complete evaluation pipelines
   - Use realistic sample data
   - Test error handling and edge cases

3. **Performance Tests**
   - Benchmark evaluation speed
   - Test memory usage
   - Validate scalability claims

## 📊 Data Contributions

### Sample Data Guidelines

1. **Data Privacy**
   - No real personal information
   - Anonymize creator profiles
   - Use synthetic engagement data

2. **Data Quality**
   - Diverse creator types and platforms
   - Realistic content examples
   - Proper data validation

3. **Data Format**
   ```json
   {
     "creator_id": "creator_001",
     "content_samples": [...],
     "engagement_data": [...],
     "evaluation_labels": [...]
   }
   ```

## 🏆 Recognition

### Contributor Levels

1. **🌟 Contributor**: Made accepted contributions
2. **🚀 Regular Contributor**: 5+ merged PRs
3. **💎 Core Contributor**: Significant feature development
4. **🎯 Maintainer**: Repository maintenance responsibilities

### Hall of Fame
Contributors will be recognized in:
- README contributor section
- Release notes
- Conference presentations
- Blog posts about the project

## ❓ Getting Help

### Communication Channels

1. **GitHub Discussions**: General questions and ideas
2. **GitHub Issues**: Bug reports and feature requests
3. **Discord Server**: Real-time community chat
4. **Email**: maintainers@creative-ai-eval.com

### Resources

- **[Development Setup Guide](docs/development-setup.md)**
- **[Architecture Overview](docs/architecture.md)**
- **[API Reference](docs/api-reference.md)**
- **[Testing Guide](docs/testing.md)**

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Every contribution, no matter how small, helps make this framework better for everyone. We appreciate your time and effort in improving creative AI evaluation!

**Questions?** Feel free to reach out through any of our communication channels. We're here to help! 🚀 