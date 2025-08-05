# Portfolio Pilot - AI-Enhanced Full Stack Portfolio Construction Platform

A full-stack web application for portfolio optimization using machine learning, classical algorithms, and custom AI-generated strategies powered by Anthropic Claude.

## Tech Stack & Skills Implemented

**Frontend:**
- Next.js 15 with TypeScript
- React with hooks and context API
- Tailwind CSS for styling
- Recharts for data visualization
- Axios for API calls
- Anthropic Claude SDK integration

**Backend:**
- FastAPI with Python
- Machine Learning models (ARIMA, LSTM, Autoformer)
- Portfolio optimization algorithms (Markowitz, GMVP, Reinforcement Learning)
- NumPy, Pandas, Scikit-learn, PyTorch for data processing

**Database & Authentication:**
- Supabase (OAuth and PostgreSQL)
- JWT token management

## Project Overview

Portfolio Pilot allows users to build and optimize investment portfolios using advanced machine learning techniques and AI-generated custom strategies. The platform combines classical portfolio theory with modern forecasting models and cutting-edge AI to help users create sophisticated, personalized investment strategies.

### Key Features

- **Pre-built Algorithms**: Choose from proven portfolio optimization and forecasting models
- **Real-time Backtesting**: Test strategies against historical data with comprehensive performance metrics
- **Interactive Visualizations**: View portfolio compositions, equity curves, and forecast charts
- **Session History**: Track and compare all your strategy experiments
  
**Home Page** - Landing page with hero section, feature showcase, algorithm overview, and call-to-action elements.
![Home Page](https://github.com/user-attachments/assets/3ad5b6ae-4790-4918-be4a-b6d085259e93)

**Dashboard** - Interactive interface for stock selection and model training initiation.
![Dashboard](https://github.com/user-attachments/assets/02a4133f-aaca-487b-b41b-cccaa347b556)

**Custom AI Strategies**: Leverage Anthropic Claude to generate bespoke investment algorithms based on natural language descriptions
![Custom AI Model](https://github.com/user-attachments/assets/1e0b5522-33b8-45ca-a543-1a14d966659c)

**History** - View past training results and portfolio performance analytics.
![History](https://github.com/user-attachments/assets/48a8fe79-23b5-40fe-961f-a682a6a00d1f)

## Project Structure

```
.
├─ frontend/             ← Next.js web application
│   ├─ app/             ← App router pages and API routes
│   ├─ components/      ← Reusable React components
│   ├─ contexts/        ← React context providers
│   └─ lib/             ← Utility functions and configs
├─ backend/             ← FastAPI server
│   ├─ models/          ← ML portfolio optimization models
│   ├─ forecasting/     ← Time series forecasting models
│   ├─ utils/           ← Helper functions and data processing
│   ├─ main.py          ← FastAPI application entry point
│   └─ requirements.txt ← Python dependencies
└─ README.md           ← This file
```

## Setup & Installation

### Backend Dependencies
```bash
cd backend
pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Frontend Dependencies
```bash
cd frontend
npm install
```

### Running the Application

**Start Backend Server:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Start Frontend Development Server:**
```bash
cd frontend
npm run dev
```

## How to Use

### Standard Algorithm Usage
![Standard](https://github.com/user-attachments/assets/a731a76a-0967-4a3c-9bcb-78a20ff75c7e)

### Custom AI Strategy Generation
![Custom AI](https://github.com/user-attachments/assets/9fae1330-6469-4382-babf-b3414b1b8754)

**Getting Started:**
1. **Sign In/Login** - Authenticate using OAuth through Supabase
2. **Dashboard** - Navigate to the main dashboard
3. **Pick Stocks** - Select up to 8 stocks from the DOW30 for your portfolio
4. **Choose Strategy Type**:
   - **Pre-built Models** - Select from classical algorithms (ARIMA, LSTM, Markowitz, etc.)
   - **Custom AI Strategy** - Describe your investment approach in natural language
5. **Train & Analyze** - Execute your strategy and view comprehensive results
6. **View History** - Review past training results and portfolio performance

## Algorithms Implemented

### Portfolio Optimization Models
- **Naive Markowitz** - Mean-variance optimization with risk-aversion parameter
- **GMVP (Global Minimum Variance Portfolio)** - Minimizes portfolio variance
- **GMVP with Clustering** - Cost-aware GMVP with stock clustering for stability
- **MarginTrader** - Reinforcement learning agent using A2C algorithm
- **Portfolio Policy Network** - Deep learning approach for portfolio allocation

### Forecasting Models
- **ARIMA** - Auto-regressive integrated moving average for time series prediction
- **LSTM** - Long short-term memory neural networks for sequence modeling
- **Autoformer** - Transformer-based model for long-term time series forecasting

### 🤖 Custom AI-Generated Strategies
- **Natural Language Processing** - Describe your investment strategy in plain English
- **Code Generation** - Anthropic Claude converts descriptions into executable TypeScript algorithms
- **Dynamic Execution** - Real-time strategy compilation and backtesting
- **Security Validation** - Multi-layer security checks ensure safe code execution

All models use consistent evaluation parameters including transaction costs, rebalancing frequency, and lookback windows for fair comparison.
