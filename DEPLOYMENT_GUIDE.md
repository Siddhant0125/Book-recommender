# 📚 Streamlit App Deployment Guide

## Option 1: Deploy on Streamlit Cloud (Recommended - Easiest)

### Prerequisites
- GitHub account
- Your project pushed to GitHub
- Streamlit account (free at https://streamlit.io)

### Steps

#### 1. Push Your Code to GitHub

```bash
cd "C:\Users\siddh\Documents\Book recommender"
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/book-recommender.git
git branch -M main
git push -u origin main
```

#### 2. Go to Streamlit Cloud

1. Visit https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - **Repository**: `YOUR_USERNAME/book-recommender`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click "Deploy"

#### 3. Configure App Settings (Optional)

- Click "Settings" → "Advanced settings"
- Set Python version, memory, etc. as needed

### ✅ That's it! Your app will be live at:
```
https://your-username-book-recommender.streamlit.app
```

**Note**: Streamlit Cloud will automatically install dependencies from `requirements.txt`

---

## Option 2: Deploy on Heroku (Free tier no longer available, but still possible with paid tier)

### Steps

1. Install Heroku CLI
2. Create `Procfile` in your project root:
   ```
   web: streamlit run streamlit_app.py --logger.level=error
   ```

3. Create `.streamlit/config.toml`:
   ```toml
   [server]
   headless = true
   port = $PORT
   ```

4. Deploy:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

---

## Option 3: Deploy on AWS EC2

### Quick Setup

1. Launch an EC2 instance (Ubuntu 22.04)
2. Connect via SSH
3. Run:
   ```bash
   sudo apt update && sudo apt install python3-pip python3-venv -y
   git clone https://github.com/YOUR_USERNAME/book-recommender.git
   cd book-recommender
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Run on port 8501
   streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
   ```

4. Configure security group to allow port 8501
5. Access at: `http://your-ec2-public-ip:8501`

---

## Option 4: Deploy on PythonAnywhere (Easiest Alternative to Streamlit Cloud)

1. Sign up at https://www.pythonanywhere.com (free account available)
2. Upload your code via Git or file upload
3. Create a web app with Flask/WSGI wrapper
4. Your app runs at: `your-username.pythonanywhere.com`

---

## Option 5: Deploy with Docker

### Create `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
docker build -t book-recommender .
docker run -p 8501:8501 book-recommender
```

### Deploy to Docker Hub / Cloud Container Services
- Push to Docker Hub and deploy on AWS ECS, Google Cloud Run, Azure Container Instances, etc.

---

## Important Considerations

### File Size Issues
Your model file `two_tower.pt` and datasets might be large. Streamlit Cloud has limitations:
- **Free tier**: 1GB storage per app
- Check file sizes:
  ```bash
  ls -lh models/
  ls -lh preprocessed/
  ls -lh dataset/
  ```

### If Files Are Too Large
1. **Option A**: Use Git LFS (Large File Storage)
   ```bash
   git lfs install
   git lfs track "*.pt" "*.csv" "*.parquet"
   git add .gitattributes
   git commit -m "Add LFS tracking"
   git push
   ```

2. **Option B**: Store files in cloud storage (S3, Google Cloud Storage) and download at runtime
   ```python
   # In recsys_backend.py, modify load_checkpoint() to fetch from S3
   ```

3. **Option C**: Use a paid Streamlit Cloud subscription for more storage

### Performance Optimization
- The app uses `@st.cache_resource` which caches the model - good! ✅
- Model loads once and is reused for all sessions

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'backend'"
**Solution**: Ensure your GitHub repo includes all directories (`backend/`, `models/`, `preprocessed/`, etc.)

### Issue: "CUDA not available" error
**Solution**: The app already handles this with:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
Streamlit Cloud uses CPU, which is fine.

### Issue: Model loads slowly
**Solution**: This is normal for first load. Streamlit caches it, so subsequent uses are instant.

### Issue: App crashes or times out
**Solution**: 
- Check if your model/data files are too large
- Use Git LFS for large files
- Consider splitting the recommendation pipeline

---

## Quick Comparison Table

| Platform | Cost | Setup Time | Cold Start | Storage | Notes |
|----------|------|-----------|-----------|---------|-------|
| **Streamlit Cloud** | Free | 5 min | ~2-3s | 1GB free | Best for demos, recommended |
| **Heroku** | $7+/month | 10 min | ~5s | Good | Reliable, but paid |
| **AWS EC2** | $2-10/month | 20 min | <1s | Flexible | Most control |
| **PythonAnywhere** | Free/$5+/month | 15 min | ~2s | Limited | Good alternative |
| **Docker + Cloud** | Varies | 30 min | <1s | Flexible | Most scalable |

---

## Recommended: Streamlit Cloud

**My recommendation** is to use **Streamlit Cloud** because:
✅ Free tier available  
✅ Easiest deployment (2 clicks)  
✅ Automatic deployments on GitHub push  
✅ Built for Streamlit apps  
✅ Good performance for this use case  
⚠️ Just make sure your model files aren't too large (check with Git LFS if needed)

---

## Next Steps

1. Create a GitHub account if you don't have one
2. Push your code to GitHub
3. Sign up for Streamlit Cloud at https://share.streamlit.io
4. Deploy in 2 clicks!

Good luck! 🚀

