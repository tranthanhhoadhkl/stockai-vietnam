# StockAI Vietnam – Deploy lên Render.com

## Cấu trúc project
```
stockai/
├── main.py              # FastAPI backend
├── requirements.txt     # Dependencies
├── render.yaml          # Cấu hình Render
└── static/
    └── index.html       # Frontend SPA
```

## Deploy lên Render (làm 1 lần)

### Bước 1 – Đẩy lên GitHub
```bash
git init
git add .
git commit -m "StockAI Vietnam - first deploy"
git branch -M main
git remote add origin https://github.com/TEN_BAN/stockai-vietnam.git
git push -u origin main
```

### Bước 2 – Tạo Web Service trên Render
1. Vào https://render.com → Đăng ký / Đăng nhập
2. Nhấn **New → Web Service**
3. Chọn **Connect a repository** → Chọn repo `stockai-vietnam`
4. Render tự đọc `render.yaml` → Nhấn **Create Web Service**

### Bước 3 – Chờ deploy (~3-5 phút)
- Render tự cài `requirements.txt`
- Tự chạy `uvicorn main:app`
- Bạn nhận được link: `https://stockai-vietnam.onrender.com`

## Lưu ý quan trọng
- **Free tier**: App sẽ sleep sau 15 phút không có request → lần đầu vào chờ ~30-50 giây
- **Upgrade**: $7/tháng để app luôn chạy (không sleep)
- Mỗi lần push code lên GitHub → Render tự động re-deploy
