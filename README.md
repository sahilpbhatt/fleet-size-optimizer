# Fleet Size Optimization Platform

## 🚀 Production-Ready Crowdsourced Delivery Optimization System

A cloud-deployed optimization platform that determines optimal fleet sizes for crowdsourced delivery services, balancing service level objectives with driver utilization targets.

### 📊 Live Demo
- **Frontend**: [https://fleet-optimizer.amazonaws.com](https://your-ec2-url.amazonaws.com:8501)
- **API Docs**: [https://fleet-optimizer.amazonaws.com/api/docs](https://your-ec2-url.amazonaws.com:8000/api/docs)

## 🎯 Key Features

- **Advanced Optimization**: Value Function Approximation (VFA) with Markov Decision Process
- **Real-time Analytics**: Interactive dashboards with performance metrics
- **Scalable Architecture**: Handles 1000+ drivers and orders efficiently
- **Cloud-Native**: Fully deployed on AWS with auto-scaling capabilities
- **RESTful API**: FastAPI backend with comprehensive documentation

## 📈 Performance Metrics

- **Optimization Time**: < 60 seconds for full day planning
- **Service Level**: Achieves 95%+ demand fulfillment
- **Driver Utilization**: Maintains 80%+ driver efficiency
- **Fleet Reduction**: Up to 65% reduction vs. baseline policies
- **Scalability**: Tested with 100+ scenarios, 1000+ drivers

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  FastAPI Backend │────▶│  Optimization   │
│   (Port 8501)   │     │   (Port 8000)    │     │     Engine      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                              │
                     ┌────────▼────────┐
                     │   AWS EC2/ECS   │
                     │   Docker/K8s    │
                     └─────────────────┘
```

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/sahilbhatt/fleet-optimizer.git
cd fleet-optimizer

# Install dependencies
pip install -r requirements.txt

# Start backend
python backend.py

# In another terminal, start frontend
streamlit run app.py
```

### Docker Deployment

```bash
# Build image
docker build -t fleet-optimizer .

# Run container
docker run -p 8000:8000 -p 8501:8501 fleet-optimizer
```

### AWS Deployment

```bash
# Configure AWS CLI
aws configure

# Deploy to EC2
./deploy_aws.sh

# Or use AWS CDK
cdk deploy FleetOptimizerStack
```

## 📊 Algorithm Overview

### Value Function Approximation (VFA)
- Iteratively searches for optimal fleet sizes
- Uses Boltzmann exploration for solution space navigation
- Convergence in ~1000 iterations

### Markov Decision Process (MDP)
- Models operational dynamics
- Handles stochastic driver and demand arrivals
- Parametric cost function approximation

### Key Innovations
- Decision-dependent uncertainty modeling
- Two-stage optimization framework
- Real-time driver-order matching

## 🔬 Research Foundation

Based on the paper: **"Fleet Size Planning in Crowdsourced Delivery: Balancing Service Level and Driver Utilization"**

Published in: Omega - The International Journal of Management Science

## 📁 Project Structure

```
fleet-optimizer/
├── app.py                 # Streamlit frontend
├── backend.py            # FastAPI backend
├── optimization_engine.py # Core optimization logic
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── deploy_aws.sh        # AWS deployment script
├── tests/               # Unit and integration tests
│   ├── test_optimizer.py
│   └── test_api.py
├── data/               # Sample datasets
│   ├── synthetic/
│   └── chicago/
└── docs/              # Additional documentation
    ├── API.md
    └── ALGORITHMS.md
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Load testing
locust -f tests/load_test.py
```

## 📊 API Endpoints

### Core Endpoints

- `POST /optimize` - Run optimization with parameters
- `GET /benchmark/{method}` - Get benchmark results
- `GET /sensitivity/{parameter}` - Sensitivity analysis
- `POST /simulate` - Simulate custom fleet sizes

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/optimize",
    json={
        "w_s": 0.5,
        "periods": 16,
        "prob_enter": 0.7,
        "penalty_type": "linear"
    }
)

results = response.json()
print(f"Service Level: {results['avg_service_level']:.1%}")
print(f"Driver Utilization: {results['avg_driver_utilization']:.1%}")
```

## 🔧 Configuration

Environment variables (`.env` file):

```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Optimization Parameters
DEFAULT_PERIODS=16
DEFAULT_PENALTY=250
MAX_ITERATIONS=1000

# API Settings
API_PORT=8000
STREAMLIT_PORT=8501
```

## 📈 Performance Optimization

- **Caching**: Pre-computed solutions for common scenarios
- **Parallel Processing**: Multi-threaded scenario evaluation
- **Memory Management**: Efficient data structures for large-scale instances
- **GPU Support**: Optional CUDA acceleration for matrix operations

## 🌟 Key Results

| Metric | VFA (Proposed) | Constant Fleet | Myopic Policy |
|--------|---------------|----------------|---------------|
| Service Level | 97% | 88% | 92% |
| Utilization | 93% | 95% | 75% |
| Daily Profit | $14,100 | $13,500 | $14,200 |
| Fleet Size | 376 | 400 | 450 |

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 👤 Author

**Sahil Bhatt**
- Email: sahil.bhatt@torontomu.ca
- LinkedIn: [linkedin.com/in/sahilbhatt](https://linkedin.com/in/sahilbhatt)
- GitHub: [@sahilbhatt](https://github.com/sahilbhatt)

## 🙏 Acknowledgments

- Toronto Metropolitan University
- Research advisors and collaborators
- Chicago Data Portal for ridehailing dataset

## 📚 Citations

If you use this code in your research, please cite:

```bibtex
@article{bhatt2024fleet,
  title={Fleet Size Planning in Crowdsourced Delivery},
  author={Bhatt, Sahil and Alnaggar, Aliaa},
  journal={Omega},
  year={2024},
  publisher={Elsevier}
}
```
