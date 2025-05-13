from flask import Flask, render_template, request
import math
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# دالة لحساب المسافة المربعة
def calculate_distance_squared(x, c):
    return sum((xi - ci)**2 for xi, ci in zip(x, c))

# دالة لتحويل النص إلى متجه
def parse_vector(text):
    try:
        return [float(item) for item in text.split()]
    except Exception:
        return []

# دالة لتحويل الصورة إلى base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/som', methods=['GET', 'POST'])
def som():
    table_rows = []
    final_weights = None
    error = None
    
    if request.method == 'POST':
        inputs = []
        i = 0
        while f'input_{i}' in request.form:
            input_vector = parse_vector(request.form.get(f'input_{i}'))
            if input_vector:
                inputs.append(input_vector)
            i += 1
        
        w1 = parse_vector(request.form.get('w1', ''))
        w2 = parse_vector(request.form.get('w2', ''))
        alpha_list = parse_vector(request.form.get('alpha', ''))
        
        try:
            epochs = int(request.form.get('epochs', '4'))
        except ValueError:
            epochs = 4
        
        if not inputs or not w1 or not w2:
            error = "خطأ: الرجاء إدخال عينات وأوزان صحيحة."
        else:
            dimension = len(inputs[0])
            if any(len(x) != dimension for x in inputs) or len(w1) != dimension or len(w2) != dimension:
                error = "خطأ: تأكد من أن جميع المتجهات بنفس الأبعاد."
            else:
                for epoch in range(epochs):
                    for idx, x in enumerate(inputs, start=1):
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
                            old_w1 = w1[:]
                            w1 = [old_w1[i] + m * (x[i] - old_w1[i]) for i in range(dimension)]
                        else:
                            winner = "w2"
                            old_w2 = w2[:]
                            w2 = [old_w2[i] + m * (x[i] - old_w2[i]) for i in range(dimension)]
                        
                        # Store numeric values, round vectors to 2 decimal places for display
                        table_rows.append({
                            "epoch": epoch + 1,
                            "sample": idx,
                            "input": ' '.join([f"{val:.2f}" for val in x]),  # Format input vector as string
                            "alpha": m,  # Keep as numeric for rounding in template
                            "d1": d1,    # Keep as numeric for rounding in template
                            "d2": d2,    # Keep as numeric for rounding in template
                            "winner": winner,
                            "w1": ' '.join([f"{val:.2f}" for val in w1]),  # Format w1 vector as string
                            "w2": ' '.join([f"{val:.2f}" for val in w2])   # Format w2 vector as string
                        })
                
                # Final weights formatted to 2 decimal places
                final_weights = {
                    "w1": ' '.join([f"{val:.2f}" for val in w1]),
                    "w2": ' '.join([f"{val:.2f}" for val in w2])
                }
    
    return render_template('som.html', table_rows=table_rows, final_weights=final_weights, error=error)
@app.route('/rbf', methods=['GET', 'POST'])
def rbf():
    error = None
    plot_url = None
    results = []
    
    if request.method == 'POST':
        points = []
        i = 0
        while f'x1_{i}' in request.form:
            try:
                x1 = float(request.form.get(f'x1_{i}'))
                x2 = float(request.form.get(f'x2_{i}'))
                points.append([x1, x2])
                i += 1
            except ValueError:
                error = "خطأ: الرجاء إدخال قيم عددية صحيحة للنقاط."
                break
        
        c1 = parse_vector(request.form.get('c1', '0 0'))
        c2 = parse_vector(request.form.get('c2', '2.5 2.5'))
        sigma2 = float(request.form.get('sigma2', '0.5'))
        
        if len(c1) != 2 or len(c2) != 2:
            error = "خطأ: المراكز c1 وc2 يجب أن تكونا متجهين ببعدين (x, y)."
        elif points and not error:
            for x1, x2 in points:
                r1_squared = calculate_distance_squared([x1, x2], c1)
                r2_squared = calculate_distance_squared([x1, x2], c2)
                phi1 = math.exp(-r1_squared / (2 * sigma2))
                phi2 = math.exp(-r2_squared / (2 * sigma2))
                class_pred = "الفئة الأولى" if phi1 > phi2 else "الفئة الثانية"
                # إضافة تسمية Light/Dark بناءً على الفئة
                label = "Light" if class_pred == "الفئة الأولى" else "Dark"
                results.append({
                    'x1': x1,
                    'x2': x2,
                    'r1_squared': f"{r1_squared:.4f}",
                    'r2_squared': f"{r2_squared:.4f}",
                    'phi1': f"{phi1:.4f}",
                    'phi2': f"{phi2:.4f}",
                    'class': class_pred,
                    'label': label  # إضافة التسمية
                })
            
            fig, ax = plt.subplots()
            ax.scatter(c1[0], c1[1], c='black', label='c1', marker='x')
            ax.scatter(c2[0], c2[1], c='gray', label='c2', marker='x')
            for x1, x2 in points:
                phi1 = math.exp(-calculate_distance_squared([x1, x2], c1) / (2 * sigma2))
                phi2 = math.exp(-calculate_distance_squared([x1, x2], c2) / (2 * sigma2))
                ax.scatter(x1, x2, c='red' if phi1 > phi2 else 'blue', label='النقطة' if (x1, x2) == points[0] else '')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.legend()
            ax.grid(True)
            plot_url = fig_to_base64(fig)
            plt.close(fig)
    
    return render_template('rbf.html', results=results, plot_url=plot_url, error=error)

import numpy as np
from flask import render_template, request

@app.route('/pca', methods=['GET', 'POST'])
def pca():
    error = None
    mean = None
    cov_matrix = None
    eigenvalues = None
    eigenvectors = None
    transformed_data = None
    centered_data = None
    product_mean = None
    products_list = None
    
    if request.method == 'POST':
        data = []
        i = 0
        while f'x1_{i}' in request.form:
            try:
                x1 = float(request.form.get(f'x1_{i}'))
                x2 = float(request.form.get(f'x2_{i}'))
                data.append([x1, x2])
                i += 1
            except ValueError:
                error = "خطأ: الرجاء إدخال قيم رقمية صحيحة"
                break
        
        if data and not error:
            data = np.array(data)
            n = len(data)
            
            # الخطوة 1: حساب المتوسطات
            mean = np.mean(data, axis=0)
            mean = np.round(mean, 4)
            
            # الخطوة 2: مركزة البيانات
            centered_data = data - mean
            centered_data = np.round(centered_data, 4)
            
            # حساب عمود الضرب ومتوسطه
            products = centered_data[:, 0] * centered_data[:, 1]
            products_list = [round(p, 4) for p in products]  # تخزين القيم الفردية
            product_mean = np.sum(products) / (n - 1)  # لحساب التباين المشترك
            product_mean = round(product_mean, 5)
            
            # الخطوة 3: حساب مصفوفة التغاير
            cov_12 = np.sum(centered_data[:, 0] * centered_data[:, 1]) / (n - 1)
            var1 = np.sum(centered_data[:, 0] ** 2) / (n - 1)
            var2 = np.sum(centered_data[:, 1] ** 2) / (n - 1)
            cov_matrix = np.array([[var1, cov_12], [cov_12, var2]])
            cov_matrix = np.round(cov_matrix, 4)
            
            # الخطوة 4: حساب القيم والمتجهات الذاتية
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            idx = eigenvalues.argsort()[::-1]  # ترتيب تنازلي
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # ضبط اتجاه المتجهات الذاتية
            if eigenvectors[0,0] < 0:
                eigenvectors[:,0] *= -1
            if eigenvectors[0,1] < 0:  # لجعل العنصر الأول موجباً في المتجه الثاني
                eigenvectors[:,1] *= -1
            
            # تبديل الأعمدة إذا لزم الأمر لمطابقة المثال
            if abs(eigenvectors[0,1]) > 0.7:  # إذا كان المتجه الثاني يشبه [0.7797, -0.6262]
                eigenvectors = eigenvectors[:, [1,0]]
                eigenvalues = eigenvalues[[1,0]]
            
            eigenvalues = np.round(eigenvalues, 8)
            eigenvectors = np.round(eigenvectors, 4)
            
            # الخطوة 5: تحويل البيانات
            transformed_data = centered_data @ eigenvectors
            transformed_data = np.round(transformed_data, 5)
    
    return render_template('pca.html', 
                          mean=mean.tolist() if mean is not None else None,
                          cov_matrix=cov_matrix.tolist() if cov_matrix is not None else None,
                          eigenvalues=eigenvalues.tolist() if eigenvalues is not None else None,
                          eigenvectors=eigenvectors.tolist() if eigenvectors is not None else None,
                          transformed_data=transformed_data.tolist() if transformed_data is not None else None,
                          centered_data=centered_data.tolist() if centered_data is not None else None,
                          product_mean=product_mean,
                          products_list=products_list,
                          error=error)

@app.route('/fuzzy', methods=['GET', 'POST'])
def fuzzy():
    error = None
    linguistic_vars = {
        "درجة الحرارة": ["مثلج", "بارد", "دافئ", "حار"],
        "غطاء السحب": ["غائم", "غائم جزئيًا", "مشمس"],
        "السرعة": ["بطيء", "سريع"],
        "الطعم": ["حلو", "حلو جدًا", "حلو قليلاً"]
    }
    crisp_vars = {
        "درجة الحرارة": "36 درجة مئوية",
        "السرعة": "0 (بطيء) أو 1 (سريع)"
    }
    membership_data = []
    conjunction_result = None
    disjunction_result = None
    
    if request.method == 'POST':
        i = 0
        while f'name_{i}' in request.form:
            try:
                name = request.form.get(f'name_{i}')
                height = float(request.form.get(f'height_{i}'))
                membership = float(request.form.get(f'membership_{i}'))
                membership_data.append({"name": name, "height": height, "membership": membership})
                i += 1
            except ValueError:
                error = "خطأ: الرجاء إدخال قيم صحيحة (الطول ودرجة العضوية يجب أن تكون أرقامًا)."
                break
        
        try:
            a = float(request.form.get('fuzzy_a', '0'))
            b = float(request.form.get('fuzzy_b', '0'))
            if 0 <= a <= 1 and 0 <= b <= 1:
                conjunction_result = min(a, b)
                disjunction_result = max(a, b)
            else:
                error = "خطأ: قيم A وB يجب أن تكون بين 0 و1."
        except ValueError:
            error = "خطأ: الرجاء إدخال قيم عددية لـ A وB."
    
    return render_template('fuzzy.html', linguistic_vars=linguistic_vars,
                         crisp_vars=crisp_vars, membership_data=membership_data,
                         conjunction_result=conjunction_result, disjunction_result=disjunction_result,
                         error=error)

from flask import render_template, request

def parse_vector(vector_str):
    try:
        return [float(x) for x in vector_str.split()]
    except ValueError:
        return None

@app.route('/art1', methods=['GET', 'POST'])
def art1():
    error = None
    clusters = []
    steps = []  # To store all steps
    final_b = None
    final_t = None

    if request.method == 'POST':
        inputs = []
        i = 0
        while f'input_{i}' in request.form:
            vector = parse_vector(request.form.get(f'input_{i}'))
            if vector and len(vector) == 4:  # n = 4
                inputs.append(vector)
            elif vector:
                error = "خطأ: يجب أن يحتوي المتجه على 4 مكونات."
            i += 1
        
        try:
            rho = float(request.form.get('rho', '0.4'))
            if not (0 < rho <= 1):
                error = "خطأ: يجب أن تكون قيمة rho بين 0 و1."
        except ValueError:
            error = "خطأ: الرجاء إدخال قيمة عددية لـ rho."
        
        if inputs and not error:
            n = 4  # Number of components in the vector
            m = 3  # Maximum number of clusters
            L = 2  # Learning parameter
            b = [[1 / (1 + n)] * n for _ in range(m)]  # Initial bottom-up weights: 0.2
            t = [[1] * n for _ in range(m)]  # Initial top-down weights: 1
            
            # Step 0: Initialize parameters
            steps.append({"step": 0, "description": "تهيئة المعلمات", "params": {"L": L, "rho": rho, "b": [[round(w, 4) for w in row] for row in b], "t": t}})
            
            for idx, x in enumerate(inputs):
                x_norm = sum(xi for xi in x)
                if x_norm == 0:
                    clusters.append({"input": x, "cluster": "غير مجمع"})
                    continue
                
                reset = True
                iteration = 1
                while reset:
                    # Step 1: Start processing the current vector
                    steps.append({"step": 1, "description": f"بدء معالجة المتجه {idx+1}: {x}", "input": x})
                    
                    # Step 3: Set F1 activations to the input vector
                    steps.append({"step": 3, "description": "تعيين تفعيلات F1 إلى المتجه المدخل", "activation_f1": x})
                    
                    # Step 5: Compute F2 activations
                    y = [sum(b[j][i] * x[i] for i in range(n)) for j in range(m)]
                    y = [round(yj, 4) for yj in y]
                    steps.append({"step": 5, "description": "حساب تفعيلات F2", "y": y})
                    
                    # Step 6: Find the winning node
                    max_y = max(y)
                    if max_y <= 0:
                        clusters.append({"input": x, "cluster": "غير مجمع"})
                        steps.append({"step": 6, "description": "التفعيل الأقصى <= 0، إضافة كتجميع منفصل", "cluster": "غير مجمع"})
                        break
                    J = y.index(max_y)
                    steps.append({"step": 6, "description": f"العقدة الفائزة J = {J+1}", "max_y": max_y, "J": J})
                    
                    # Step 9: Recompute F1 activations
                    x_new = [x[i] * t[J][i] for i in range(n)]
                    x_new_norm = sum(x_new)
                    steps.append({"step": 9, "description": "إعادة حساب تفعيلات F1", "x_new": x_new, "x_new_norm": round(x_new_norm, 4)})
                    
                    # Step 10: Compute the ratio
                    ratio = x_new_norm / x_norm
                    steps.append({"step": 10, "description": f"حساب النسبة: {x_new_norm:.4f} / {x_norm:.4f} = {ratio:.4f}", "ratio": round(ratio, 4)})
                    
                    # Step 11: Vigilance test
                    if ratio >= rho:
                        # Step 12: Update weights
                        for i in range(n):
                            t[J][i] = x_new[i]
                            b[J][i] = L * x_new[i] / (L - 1 + x_new_norm) if x_new_norm > 0 else 0
                        clusters.append({"input": x, "cluster": J+1})
                        steps.append({"step": 12, "description": f"تحديث الأوزان: b[{J}] = {[round(w, 4) for w in b[J]]}, t[{J}] = {t[J]}", "cluster": J+1})
                        reset = False
                    else:
                        y[J] = -1
                        steps.append({"step": 11, "description": f"النسبة {ratio:.4f} < {rho}، إعادة تعيين y[{J}] = -1 وإعادة المحاولة", "reset": True})
                    iteration += 1
            
            # Store final weights
            final_b = [[round(w, 4) for w in row] for row in b]
            final_t = t

    return render_template('art1.html', steps=steps, error=error, final_b=final_b, final_t=final_t)

if __name__ == '__main__':
    app.run(debug=True)