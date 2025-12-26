# Contributing to Fish Health Monitoring System

Thank you for your interest in contributing to the Fish Health Monitoring System! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU info)
- Relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach
- Any potential drawbacks

### Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Test your changes** thoroughly:
   ```bash
   pytest tests/
   ```

4. **Update documentation** if needed

5. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/videos if applicable

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/Fish-CNN.git
   cd Fish-CNN
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install dev dependencies
   ```

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

Format code with Black:
```bash
black src/ examples/ tests/
```

Check style with Flake8:
```bash
flake8 src/ examples/ tests/
```

### Documentation

- Use Google-style docstrings
- Document all public functions, classes, and modules
- Include examples in docstrings where helpful
- Update README.md for new features

Example docstring:
```python
def analyze_health(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> HealthMetrics:
    """
    Analyze fish health from image region.

    Args:
        frame: Input image frame (BGR format)
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        HealthMetrics object with health scores

    Raises:
        ValueError: If bbox is invalid

    Examples:
        >>> metrics = analyze_health(frame, (10, 10, 100, 100))
        >>> print(metrics.overall_health_score)
        0.85
    """
```

### Commit Messages

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build/tooling changes

Example:
```
feat: add species-specific health thresholds

- Add configuration for species-specific parameters
- Update health analyzer to use species thresholds
- Add tests for new functionality

Closes #123
```

## Testing

### Writing Tests

- Write tests for new functionality
- Maintain or improve code coverage
- Use pytest framework
- Test edge cases and error conditions

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

Run specific test:
```bash
pytest tests/test_detector.py::test_detection
```

## Project Structure

```
Fish-CNN/
├── src/                    # Source code
│   ├── detection/         # Detection module
│   ├── tracking/          # Tracking module
│   ├── health_assessment/ # Health assessment
│   ├── behavior_analysis/ # Behavior analysis
│   ├── alerts/           # Alert system
│   └── utils/            # Utilities
├── examples/              # Example scripts
├── tests/                 # Unit tests
├── configs/              # Configuration files
├── docs/                 # Additional documentation
└── deployment/           # Deployment configs
```

## Areas for Contribution

We especially welcome contributions in:

1. **Species-Specific Models**: Health models for specific fish species
2. **Deep Learning**: Improved health classification models
3. **Performance**: Optimization and speed improvements
4. **Documentation**: Tutorials, guides, and examples
5. **Testing**: More comprehensive test coverage
6. **Features**: See GitHub issues labeled "enhancement"

## Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Focus on the problem, not the person
- Welcome newcomers and help them contribute
- Give credit where it's due

## Questions?

- Check existing issues and discussions
- Ask in GitHub Discussions
- Tag maintainers for guidance

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
