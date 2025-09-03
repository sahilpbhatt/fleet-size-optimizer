# 🚚 Fleet Size Optimization Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fleet-size-optimizer-fvoej2nvaszyu2uepjqshr.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Omega%20Journal-purple.svg)](https://doi.org/journal.omega)

## 🎯 Production-Ready Optimization for Crowdsourced Delivery

A cloud-deployed platform that determines optimal fleet sizes for crowdsourced delivery services, balancing service level objectives with driver utilization targets using advanced optimization techniques.

### 🚀 [Live Demo](https://fleet-size-optimizer-fvoej2nvaszyu2uepjqshr.streamlit.app/)

## 📊 Key Results

| Metric | Value | Impact |
|--------|-------|--------|
| Fleet Reduction | **65%** | vs. baseline policy |
| Service Level | **97%** | exceeds 95% target |
| Driver Utilization | **93%** | exceeds 80% target |
| Optimization Time | **<60s** | real-time capable |

## 🔬 Research Foundation

Based on the paper: **"Fleet Size Planning in Crowdsourced Delivery: Balancing Service Level and Driver Utilization"**
- Authors: Aliaa Alnaggar, Sahil Bhatt
- Journal: Omega - The International Journal of Management Science
- Status: Under Review (2024)

## ⚡ Quick Start

### Option 1: Use Live Demo
Visit [https://fleet-size-optimizer-fvoej2nvaszyu2uepjqshr.streamlit.app/](https://fleet-size-optimizer-fvoej2nvaszyu2uepjqshr.streamlit.app/)

### Option 2: Run Locally
```bash
# Clone repository
git clone https://github.com/sahilpbhatt/fleet-size-optimizer.git
cd fleet-size-optimizer

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
```

### Option 3: Docker
```bash
docker pull sahilbhatt/fleet-optimizer
docker run -p 8501:8501 sahilbhatt/fleet-optimizer
```

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │  ← Interactive Dashboard
└────────┬────────┘
         │
┌────────▼────────┐
│  Optimization   │  ← VFA Algorithm
│     Engine      │  ← MDP Simulation
└────────┬────────┘
         │
┌────────▼────────┐
│  Pre-computed   │  ← Research Results
│    Results      │  ← Chicago Dataset
└─────────────────┘
```

## 📈 Features

### Optimization Capabilities
- **Value Function Approximation (VFA)** with Boltzmann exploration
- **Markov Decision Process (MDP)** for operational dynamics
- **Multi-objective optimization** balancing service and utilization
- **Sensitivity analysis** across key parameters

### Interactive Dashboard
- Real-time optimization with progress tracking
- Interactive visualizations using Plotly
- Period-by-period fleet allocation
- Performance metrics and KPIs

### Professional Deployment
- Cloud-native architecture
- Responsive design for all devices
- PDF report generation
- CSV data export

## 🔧 Technical Stack

- **Backend**: Python 3.9+, NumPy, SciPy, Pandas
- **Frontend**: Streamlit, Plotly
- **Optimization**: Custom VFA implementation
- **Deployment**: Streamlit Cloud (free tier)

## 📊 Algorithm Overview

### Two-Stage Optimization Model

**Stage 1: Tactical Planning**
```python
min Σ(fleet_cost) + penalty_violations
s.t. service_level ≥ 95%
     utilization ≥ 80%
```

**Stage 2: Operational MDP**
```python
State: (drivers, orders, metrics)
Action: matching_decisions
Transition: stochastic_arrivals
Reward: profit - penalties
```

## 🎯 Use Cases

1. **Food Delivery Platforms** - Optimize driver pools for meal delivery
2. **Grocery Delivery** - Balance fleet size with demand patterns
3. **Package Delivery** - Last-mile optimization for e-commerce
4. **Ridesharing** - Adapt to ride-hailing scenarios

## 📝 File Structure

```
fleet-optimizer/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── precomputed_results.json  # Research results
├── README.md                 # Documentation
└── .streamlit/
    └── config.toml          # App configuration
```

## 🚀 Deployment Guide

### Streamlit Cloud (Recommended - Free)

1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click
4. Share your URL

### Alternative Platforms

- **Render**: `render.yaml` included
- **Railway**: `railway.json` included
- **Heroku**: `Procfile` included
- **Google Cloud Run**: `cloudbuild.yaml` included

## 📈 Performance Benchmarks

| Method | Service Level | Utilization | Fleet Size | Profit |
|--------|--------------|-------------|------------|--------|
| **VFA (Ours)** | 97% | 93% | 376 | $14,050 |
| Constant | 88% | 95% | 400 | $13,500 |
| Myopic | 92% | 75% | 450 | $14,200 |
| Greedy | 85% | 70% | 500 | $12,800 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{alnaggar2024fleet,
  title={Fleet Size Planning in Crowdsourced Delivery: 
         Balancing Service Level and Driver Utilization},
  author={Alnaggar, Aliaa and Bhatt, Sahil},
  journal={Omega},
  year={2024},
  publisher={Elsevier}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Sahil Bhatt**
- Email: sahil.bhatt@torontomu.ca
- LinkedIn: [linkedin.com/in/sahilpbhatt](https://linkedin.com/in/sahilpbhatt)
- GitHub: [@sahilpbhatt](https://github.com/sahilpbhatt)

## 🙏 Acknowledgments

- Toronto Metropolitan University
- Chicago Data Portal for ridehailing dataset
- Streamlit team for the hosting platform

---

**Note**: This is a research implementation demonstrating production deployment of academic optimization algorithms. The platform uses pre-computed results from extensive research for demonstration purposes.
