from flask import Flask, render_template, request
import math

app = Flask(__name__)

def calculate_distance_squared(x, w):
    """حساب d = sum((x-w)^2) لكل مكون"""
    return sum((xi - wi)**2 for xi, wi in zip(x, w))

def parse_vector(text):
    """تحويل نص مثل "0 0 0 0" إلى قائمة من القيم العشرية"""
    try:
        return [float(item) for item in text.split()]
    except Exception:
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    table_rows = []
    final_weights = None
    error = None
    if request.method == 'POST':
        # قراءة قيم العينات
        i1 = parse_vector(request.form.get('i1', ''))
        i2 = parse_vector(request.form.get('i2', ''))
        i3 = parse_vector(request.form.get('i3', ''))
        i4 = parse_vector(request.form.get('i4', ''))
        
        # قراءة الأوزان الابتدائية
        w1 = parse_vector(request.form.get('w1', ''))
        w2 = parse_vector(request.form.get('w2', ''))
        
        # قراءة قائمة معاملات التعلم (α)
        alpha_list = parse_vector(request.form.get('alpha', ''))
        
        try:
            epochs = int(request.form.get('epochs', '4'))
        except ValueError:
            epochs = 4
        
        inputs = [i1, i2, i3, i4]
        dimension = len(i1)
        # التحقق من أبعاد المتجهات
        if any(len(x) != dimension for x in inputs) or len(w1) != dimension or len(w2) != dimension:
            error = "خطأ: تأكد من أن جميع المتجهات بنفس الأبعاد."
        else:
            # تنفيذ التدريب لكل دورة ولكل عينة
            for epoch in range(epochs):
                for idx, x in enumerate(inputs, start=1):
                    # اختيار معامل التعلم للعينة الحالية:
                    if len(alpha_list) == 1:
                        m = alpha_list[0]
                    elif len(alpha_list) >= len(inputs):
                        m = alpha_list[idx-1]
                    else:
                        m = alpha_list[idx-1] if idx-1 < len(alpha_list) else alpha_list[-1]
                    
                    d1 = calculate_distance_squared(x, w1)
                    d2 = calculate_distance_squared(x, w2)
                    
                    if d1 < d2:
                        winner = "w1"
                        old_w1 = w1[:]  # حفظ القيمة القديمة لـ w1
                        w1 = [old_w1[i] + m * (x[i] - old_w1[i]) for i in range(dimension)]
                    else:
                        winner = "w2"
                        old_w2 = w2[:]
                        w2 = [old_w2[i] + m * (x[i] - old_w2[i]) for i in range(dimension)]
                    
                    table_rows.append({
                        "epoch": epoch + 1,
                        "sample": idx,
                        "input": ' '.join(map(str, x)),
                        "alpha": m,
                        "d1": f"{d1:.3f}",
                        "d2": f"{d2:.3f}",
                        "winner": winner,
                        "w1": ' '.join(map(str, w1)),
                        "w2": ' '.join(map(str, w2))
                    })
            final_weights = {"w1": ' '.join(map(str, w1)), "w2": ' '.join(map(str, w2))}
    return render_template('index.html', table_rows=table_rows, final_weights=final_weights, error=error)

if __name__ == '__main__':
    app.run(debug=True)
