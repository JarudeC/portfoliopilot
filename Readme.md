# Portfolio Pilot - Full Stack Portfolio Construction Platform

A full-stack web application for portfolio optimization using machine learning and classical algorithms.

## Tech Stack & Skills Implemented

**Frontend:**
- Next.js 14 with TypeScript
- React with hooks and context API
- Tailwind CSS for styling
- Recharts for data visualization
- Axios for API calls

**Backend:**
- FastAPI with Python
- Machine Learning models (ARIMA, LSTM, Autoformer)
- Portfolio optimization algorithms (Markowitz, GMVP, Reinforcement Learning)
- NumPy, Pandas, Scikit-learn for data processing

**Database & Authentication:**
- Supabase (OAuth and PostgreSQL)
- JWT token management

**Deployment:**
- Uvicorn ASGI server
- Environment configuration with python-dotenv

## Project Overview

Portfolio Pilot allows users to build and optimize investment portfolios using advanced machine learning techniques. The platform combines classical portfolio theory with modern forecasting models to help users make informed investment decisions.

**Dashboard** - Interactive interface for stock selection and model training initiation.
![dashboard](https://github.com/user-attachments/assets/02a4133f-aaca-487b-b41b-cccaa347b556)

**History** - View past training results and portfolio performance analytics.
![history](https://github.com/user-attachments/assets/48a8fe79-23b5-40fe-961f-a682a6a00d1f)

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

Uploading PortfolioPilot_Demo.mp4…

1. **Sign In/Login** - Authenticate using OAuth through Supabase
2. **Dashboard** - Navigate to the main dashboard
3. **Pick Stocks** - Select stocks for your portfolio
4. **Train Models** - Choose between forecasting or portfolio optimization training
5. **View History** - Review past training results and portfolio performance

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

All models use consistent evaluation parameters including transaction costs, rebalancing frequency, and lookback windows for fair comparison.
