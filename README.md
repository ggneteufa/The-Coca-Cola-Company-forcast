# The-Coca-Cola-Company-forcast
ระบบพยากรณ์ราคาหุ้นแบบครบวงจร (end-to-end pipeline) ด้วย LSTM (Bidirectional, TensorFlow/Keras) จากข้อมูลราคาหุ้นรายวัน (OHLCV) ที่ดึงผ่าน yfinance ออกแบบให้ใช้ได้กับหุ้นสหรัฐฯ ตัวใดก็ได้ สร้างฟีเจอร์เชิงเทคนิค (RSI, MACD, Bollinger Bands, volatility, lagged returns) พร้อมฟีเจอร์ดัชนีตลาดรวม (SPY) และปรับเป้าหมายการทำนายจาก "ราคา" เป็น "% การเปลี่ยนแปลงราคารายวัน (return)" เพื่อหลีกเลี่ยง bias จากการทำนายแบบจำค่าเดิมซ้ำ ทำ Walk-Forward Cross-Validation (แบบ expanding-window หลาย fold) ร่วมกับ ensemble averaging และโมเดลแบบ L2-regularized เพื่อลดความแปรปรวนของผลลัพธ์ และตรวจสอบผลเทียบกับ Naive Baseline ด้วย binomial significance test เพื่อประเมินความสามารถในการทายทิศทางอย่างเข้มงวด

วิธีใช้งาน:

bash
    pip install -r requirements.txt
    python lstm_stock_predictor.py

เปลี่ยนหุ้นได้ที่ STOCK_TICKER บรรทัดเดียว กราฟผลลัพธ์จะถูกเซฟไว้ที่ lstm_stock_forecast.png ในโฟลเดอร์เดียวกับสคริปต์
