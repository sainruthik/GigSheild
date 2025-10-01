# 🚀 GigShield Dashboard - Deployment Guide

This guide will help you deploy your GigShield Risk Monitor dashboard online for free.

## ☁️ Option 1: Deploy to Render (Recommended - FREE)

Render offers free hosting for web services and is perfect for this project.

### Step-by-Step Instructions:

1. **Create a GitHub Account** (if you don't have one)
   - Go to https://github.com and sign up

2. **Push Your Code to GitHub**
   ```bash
   cd c:\Users\sainr\Desktop\New_Project\try
   git init
   git add .
   git commit -m "Initial commit - GigShield Risk Monitor"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/gigshield-risk-monitor.git
   git push -u origin main
   ```

3. **Sign Up for Render**
   - Go to https://render.com
   - Click "Get Started for Free"
   - Sign up with your GitHub account

4. **Deploy the Dashboard**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: gigshield-dashboard
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn src.app_server:app --host 0.0.0.0 --port $PORT`
   - Click "Create Web Service"

5. **Deploy the Data Pipeline (Optional)**
   - Click "New +" → "Background Worker"
   - Select same repository
   - Configure:
     - **Name**: gigshield-pipeline
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python run_pipeline.py`
   - Click "Create Background Worker"

6. **Access Your Dashboard**
   - Render will give you a URL like: `https://gigshield-dashboard.onrender.com`
   - Your dashboard is now live! 🎉

---

## ☁️ Option 2: Deploy to Railway (Also FREE)

Railway is another great free option with easy deployment.

### Step-by-Step Instructions:

1. **Sign Up for Railway**
   - Go to https://railway.app
   - Click "Login with GitHub"

2. **Create New Project**
   - Click "New Project"
   - Choose "Deploy from GitHub repo"
   - Select your repository

3. **Configure Services**
   Railway will auto-detect your `Procfile` and deploy both:
   - Web service (dashboard)
   - Worker service (data pipeline)

4. **Get Your URL**
   - Click on your web service
   - Click "Settings" → "Generate Domain"
   - Your dashboard is live! 🎉

---

## ☁️ Option 3: Deploy to Heroku

Heroku is a classic option (free tier available with verification).

### Step-by-Step Instructions:

1. **Install Heroku CLI**
   - Download from https://devcenter.heroku.com/articles/heroku-cli

2. **Login and Create App**
   ```bash
   heroku login
   heroku create gigshield-dashboard
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

4. **Scale Worker (Optional)**
   ```bash
   heroku ps:scale worker=1
   ```

5. **Open Your Dashboard**
   ```bash
   heroku open
   ```

---

## 🎯 What Gets Deployed

Your deployment includes:

1. **Web Dashboard** (Port 8000)
   - Real-time transaction monitoring
   - Risk predictions with explanations
   - Interactive filters and detailed views

2. **Data Pipeline** (Background Worker)
   - Transaction feeder (generates demo data)
   - Prediction engine (ML model processing)

3. **Persistent Data**
   - Predictions log (CSV file)
   - Model artifacts (scaler, encoder, trained model)

---

## ⚙️ Important Notes

### Free Tier Limitations:
- **Render**: 750 hours/month free, sleeps after 15 min of inactivity
- **Railway**: $5 free credit/month
- **Heroku**: 1000 dyno hours/month (with credit card verification)

### Wake Up Time:
Free tier services may "sleep" when inactive. First request may take 30-60 seconds to wake up.

### Data Persistence:
Free tiers have ephemeral storage. Data resets on restart. For persistent data:
- Use external database (PostgreSQL on Render/Railway/Heroku)
- Or use cloud storage (AWS S3, Google Cloud Storage)

---

## 🔧 Environment Variables (Optional)

If you want to configure your deployment:

```bash
# On Render/Railway/Heroku dashboard:
PYTHON_VERSION=3.10.0
PORT=8000  # Usually auto-set
```

---

## 📊 Monitoring Your Deployment

### Check Logs:
- **Render**: Dashboard → "Logs" tab
- **Railway**: Project → Service → "Deployments" → "View Logs"
- **Heroku**: `heroku logs --tail`

### Health Check:
Visit: `https://your-app-url.com/api/health`
Should return: `{"ok":true}`

---

## 🆘 Troubleshooting

### Dashboard shows blank page:
1. Check browser console (F12) for errors
2. Verify `/api/health` endpoint works
3. Check deployment logs for Python errors

### No new transactions appearing:
1. Verify worker/pipeline service is running
2. Check worker logs for errors
3. Ensure predictions_log.csv has recent entries

### Build failures:
1. Ensure all files are committed to git
2. Check requirements.txt has all dependencies
3. Verify Python version compatibility (3.10+)

---

## 🎉 Success!

Once deployed, share your dashboard URL with anyone!

Example: `https://gigshield-dashboard.onrender.com`

The dashboard will show:
- ✅ Real-time risk predictions
- ✅ Transaction details
- ✅ Risk factor explanations
- ✅ Interactive filtering
- ✅ Auto-refreshing data

---

## 📝 Next Steps (Optional)

1. **Custom Domain**: Add your own domain name
2. **Database**: Add PostgreSQL for persistent data
3. **Authentication**: Add login/password protection
4. **Analytics**: Track dashboard usage
5. **Alerts**: Email/SMS notifications for high-risk transactions

Happy hosting! 🚀
