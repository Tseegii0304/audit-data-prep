"""
tab_descriptions.py — Аудитын ХОУ прототип v3.4
Бүх техникийн нэр томъёог монгол тайлбартай, алгоритмуудын тайлбар нэмсэн.
"""
import streamlit as st

class TabDescriptions:

    def show_summary_description(self, n_accounts=0, n_transactions=0, n_entries=0):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f4fd 0%, #f0f7ff 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #1976D2; margin-bottom: 20px;">
            <h4 style="color: #1565C0; margin-top: 0;">📊 Нэгтгэл — Шинжилгээний ерөнхий тойм</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7;">
                Энэ таб нь шинжилгээнд хамрагдсан <b>бүх өгөгдлийн ерөнхий тоймыг</b> харуулна.
                Уламжлалт MUS (Мөнгөний нэгжид суурилсан түүвэрлэлт) арга нь нийт дансны
                зөвхөн ~20%-ийг шалгадаг бол ХОУ загвар нь <b>100% дансыг бүрэн хамарна</b>.
            </p>
            <p style="color: #555; font-size: 13px; margin-bottom: 0;">
                📌 <i>ISA 300 «Аудитын ерөнхий стратеги» стандартын дагуу аудитын хүрээг тодорхойлоход суурь мэдээлэл болно.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("ℹ️ Үзүүлэлтүүдийн тайлбар", expanded=False):
            st.markdown(f"""
| Үзүүлэлт | Утга | Аудитын ач холбогдол |
|-----------|------|---------------------|
| **Нийт данс** | {n_accounts:,} | Шинжилгээнд хамрагдсан бүх дансны тоо |
| **Гүйлгээний нийт дүн** | Дебит + Кредит | Байгууллагын санхүүгийн үйл ажиллагааны цар хүрээ |
| **ЕЖ мөр** | {n_transactions:,} | Ерөнхий журналын бие даасан гүйлгээний тоо |
| **Дансны бичилтийн тоо** | {n_entries:,} | Харилцагч×Данс×Сар хосуудын тоо |
            """)

    def show_summary_interpretation(self, stats_dict=None):
        st.markdown("""
        ---
        <div style="background-color: #FFF8E1; padding: 15px; border-radius: 8px; border-left: 4px solid #FFA000;">
            <b>💡 Аудиторт:</b> Дээрх үзүүлэлтүүд нь өгөгдлийн бүрэн бүтэн байдлыг харуулна.
            Дансны тоо огцом өөрчлөгдсөн бол бүтцийн өөрчлөлт (нэгтгэл, хуваагдал, шинэ данс) байгаа эсэхийг шалгана.
        </div>
        """, unsafe_allow_html=True)

    def show_anomaly_description(self):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fce4ec 0%, #fff5f5 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #c62828; margin-bottom: 20px;">
            <h4 style="color: #b71c1c; margin-top: 0;">🔍 Хэвийн бус данс илрүүлэлт</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7;">
                <b>Аномали (хэвийн бус байдал)</b> гэдэг нь бусад данснуудын хэв маягаас
                мэдэгдэхүйц ялгаатай зан төлөв үзүүлж буй дансуудыг хэлнэ. Гурван бие даасан
                алгоритмаар илрүүлж, <b>нэгдсэн санал нэгтгэл (Ensemble)</b> аргаар эцсийн шийдвэр гаргана.
            </p>
            <p style="color: #555; font-size: 13px; margin-bottom: 0;">
                📌 <i>ISA 315 «Эрсдэлийг тодорхойлох» болон ISA 240 «Залилангийн эрсдэл» стандартуудтай нийцнэ.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🤖 Гурван алгоритмын тайлбар — хэвийн бус данс хэрхэн илрүүлдэг вэ?", expanded=False):
            st.markdown("""
#### 1️⃣ Isolation Forest — Тусгаарлалтын ой
**Зарчим:** Мод бүтэцтэй загвар нь өгөгдлийг санамсаргүйгээр хуваана.
Хэвийн данс олон удаа хуваагдсаны эцэст тусгаарлагддаг бол **хэвийн бус данс
цөөн хуваалтаар тусгаарлагддаг** — бусдаас ялгаатай шинж чанартай учраас.

*Жишээ:* 9,909 данснаас 100 данс нь бусдаас огцом өөр → 2-3 хуваалтаар тусгаарлагдана.

| Параметр | Утга | Тайлбар |
|----------|------|---------|
| contamination | 0.10 (10%) | Нийт дансны хэдэн хувийг хэвийн бус гэж үзэх |
| n_estimators | 200 | Ашиглах модны тоо (олон = илүү нарийвчлал) |

---
#### 2️⃣ Z-score — Стандарт хазайлтын оноо
**Зарчим:** Данс бүрийн шинж чанаруудыг бүх дансны **дундаж утгатай харьцуулна**.
Дундажаас 2 стандарт хазайлтаас (2σ) их зөрсөн данс = хэвийн бус.

*Жишээ:* Дундаж эргэлт 500сая₮, стандарт хазайлт 200сая₮ →
900сая₮+ (500+2×200) эргэлттэй данс = хэвийн бус.

---
#### 3️⃣ Turn ratio — Дебит-кредит эргэлтийн харьцаа
**Зарчим:** Дансны дебит/кредит харьцааг тооцно. **95-р хувийн босго (P95)**
давсан данс = нэг чиглэлтэй гүйлгээ давамгайлж буй хэвийн бус данс.

*Жишээ:* Дебит=800сая₮, кредит=100сая₮ → харьцаа=8.0. Бусдын 95% нь 3.0-оос доош → хэвийн бус.

---
#### 🔗 Нэгдсэн санал нэгтгэл (Ensemble)
Гурван аргын үр дүнг нэгтгэж **эцсийн шийдвэр** гаргана:

| Нөхцөл | Үр дүн |
|---------|--------|
| Тусгаарлалтын ой = ✅ | ✅ Хэвийн бус |
| Z-score = ✅ **БА** Эргэлтийн харьцаа = ✅ | ✅ Хэвийн бус |
| Зөвхөн нэг арга = ✅ | ❌ Хэвийн |
            """)
        with st.expander("📊 Тархалтын график хэрхэн унших вэ?", expanded=False):
            st.markdown("""
**Цэгэн тархалтын график** нь данс бүрийг нэг цэгээр дүрсэлнэ:
- **X тэнхлэг:** Баримт дебит гүйлгээний хэмжээ (логарифм)
- **Y тэнхлэг:** Он дамнасан үлдэгдлийн өөрчлөлт (логарифм)
- 🔵 Цэнхэр = Хэвийн данс, 🔴 Улаан = Хэвийн бус данс

Хулганаа аваачихад (hover) дансны нарийвчилсан мэдээлэл гарна.
            """)

    def show_anomaly_interpretation(self, n_if=0, n_zscore=0, n_turn=0, n_ensemble=0):
        st.markdown(f"""
        ---
        <div style="background-color: #FFF8E1; padding: 15px; border-radius: 8px; border-left: 4px solid #FFA000;">
            <b>💡 Дүгнэлт:</b><br>
            • <b>Тусгаарлалтын ой</b> — {n_if} данс (олон хэмжээст огторгуйд тусгаарлагдсан)<br>
            • <b>Стандарт хазайлт</b> — {n_zscore} данс (дундажаас 2σ+ зөрсөн)<br>
            • <b>Эргэлтийн харьцаа</b> — {n_turn} данс (дебит/кредит P95 давсан)<br>
            • <b>Нэгдсэн дүн</b> — <b>{n_ensemble}</b> хэвийн бус данс<br><br>
            📌 Олон арга давхацсан дансууд хамгийн өндөр эрсдэлтэй — <b>нэн тэргүүнд</b> шалгана.
        </div>
        """, unsafe_allow_html=True)

    def show_ai_vs_mus_description(self):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f1 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #2e7d32; margin-bottom: 20px;">
            <h4 style="color: #1b5e20; margin-top: 0;">⚖️ ХОУ загвар ба уламжлалт аргын харьцуулалт</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7;">
                Хиймэл оюуны гурван хяналттай сургалтын загварын гүйцэтгэлийг
                уламжлалт <b>MUS 20% түүвэрлэлттэй</b> харьцуулна.
            </p>
            <p style="color: #555; font-size: 13px; margin-bottom: 0;">
                📌 <i>ISA 200: <b>AR = IR × CR × DR</b> (Аудитын эрсдэл = Байгаа эрсдэл × Хяналтын эрсдэл × Илрүүлэлтийн эрсдэл).
                Илрүүлэлтийн эрсдэл (DR)-ийг бууруулах нь аудитын найдвартай байдлыг нэмэгдүүлнэ.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🤖 Гурван хяналттай сургалтын загварын тайлбар", expanded=False):
            st.markdown("""
#### 1️⃣ Random Forest — Санамсаргүй ой
Олон шийдвэрийн мод (200 ширхэг) бүтээж, тус бүрийг өгөгдлийн санамсаргүй хэсэг дээр
сургана. Эцсийн шийдвэрийг **бүх модны олонхын санал** нэгтгэж гаргана.
*Давуу тал:* Өндөр нарийвчлал, шинж чанарын ач холбогдлыг тооцох боломжтой.

#### 2️⃣ Gradient Boosting — Градиентийн нэмэгдүүлэлт
Моднуудыг **дараалан** бүтээж, өмнөх модны алдааг дараагийн мод засна.
*Давуу тал:* Нарийн хэв маягийг олох чадвар хамгийн сайн. XAI тайлбарт ашигладаг.

#### 3️⃣ Logistic Regression — Логистик регресс
Шугаман тэгшитгэлээр данс бүрийн хэвийн бус байх **магадлалыг** (0-100%) тооцно.
*Давуу тал:* Хамгийн энгийн, ойлгомжтой загвар.

#### 🔄 5 давхар хөндлөн баталгаажуулалт (Cross-validation)
Өгөгдлийг 5 хэсэгт хуваана → 4-р сургаж 1-р шалгана → 5 удаа давтана.
Ингэснээр загварын найдвартай байдлыг баталгаажуулна.
            """)
        with st.expander("📏 Гүйцэтгэлийн хэмжүүрүүдийн тайлбар", expanded=False):
            st.markdown("""
| Хэмжүүр | Монгол нэр | Аудитын утга |
|----------|-----------|-------------|
| **Precision** | Нарийвчлал | Хэвийн бус гэсэн данснаас хэдэн хувь нь үнэхээр хэвийн бус вэ? |
| **Recall** | Бүрэн илрүүлэлт | Бүх хэвийн бус данснаас хэдэн хувийг зөв олсон бэ? ⚠️ **Хамгийн чухал!** |
| **F1-score** | Тэнцвэржүүлсэн оноо | Нарийвчлал ба бүрэн илрүүлэлтийн тэнцвэржүүлсэн дундаж |
| **AUC-ROC** | Муруйн доорх талбай | Загварын ялгах чадвар (1.0 = төгс) |
| **Detection Risk** | Илрүүлэлтийн эрсдэл | ⚠️ Алдаа илрэлгүй үлдэх магадлал = **1 − Recall** |

**ROC муруй:** Муруй зүүн дээд булан руу ойр = загвар сайн. Ташуу шугам = санамсаргүй таамаглал.
            """)

    def show_ai_vs_mus_interpretation(self, rf_f1=0, rf_auc=0, dr_ai="", dr_mus="", mcnemar_chi2=0):
        st.markdown(f"""
        ---
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 4px solid #2E7D32;">
            <b>📋 Дүгнэлт:</b><br>
            • Шилдэг загвар: <b>F1={rf_f1}, AUC={rf_auc}</b><br>
            • Илрүүлэлтийн эрсдэл: ХОУ <b>{dr_ai}</b> ↔ MUS <b>{dr_mus}</b> → <b>15-20 дахин бууруулсан</b><br>
            • Хугацаа: ХОУ <b>4.5 цаг</b> (100%% данс) ↔ MUS <b>310-357 цаг</b> (20%% данс) → <b>98.5%% хэмнэлт</b>
        </div>
        """, unsafe_allow_html=True)

    def show_xai_description(self):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #faf5fc 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #7b1fa2; margin-bottom: 20px;">
            <h4 style="color: #6a1b9a; margin-top: 0;">🧠 Тайлбарлагдах хиймэл оюун (XAI)</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7;">
                <b>XAI (Explainable AI)</b> буюу тайлбарлагдах хиймэл оюун нь загвар
                <b>«яагаад тухайн дансыг хэвийн бус гэж үнэлсэн бэ?»</b> гэсэн асуултад хариулна.
                Шинж чанар бүрийн нөлөөг <b>Feature Importance (шинж чанарын ач холбогдол)</b> шинжилгээгээр хэмжинэ.
            </p>
            <p style="color: #555; font-size: 13px; margin-bottom: 0;">
                📌 <i>ISA 500 «Аудитын нотолгоо» — загвар нь «хар хайрцаг» биш, шийдвэр бүр тайлбартай.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🤖 Feature Importance шинжилгээ хэрхэн ажилладаг вэ?", expanded=False):
            st.markdown("""
**Feature Importance (Шинж чанарын ач холбогдол)** нь Gradient Boosting загвараас тооцогдоно.

**Зарчим:** Загвар мод бүтээхдээ аль шинж чанараар хуваахад **алдаа хамгийн их буурах** вэ гэдгийг тооцно.
Олон удаа ашигласан, алдааг их бууруулсан шинж чанар = **өндөр ач холбогдолтой**.

**Утга:** 0.0 (нөлөөгүй) → 1.0 (бүрэн нөлөөтэй). Бүх шинж чанарын нийлбэр = 1.0

*Жишээ:* «Дансны цэвэр өөрчлөлт» хамгийн чухал шинж чанар болсон нь
ISA 520 «Аналитик горим»-тай шууд нийцнэ.
            """)

    FEATURE_DESCRIPTIONS = {
        "log_abs_change": {"mn_name": "Он дамнасан цэвэр өөрчлөлт", "description": "Дансны үлдэгдэл өмнөх жилээс хэр их өөрчлөгдсөнийг хэмжинэ. Огцом өөрчлөлт = бүртгэлийн алдаа, манипуляцийн шинж.", "isa_ref": "ISA 520 «Аналитик горим» — он дамнасан өөрчлөлтийг заавал шинжилнэ", "example": "Жишээ: Авлага 500сая₮-өөс 2.5тэрбум₮ болж 5 дахин өссөн", "icon": "📈"},
        "turn_ratio": {"mn_name": "Дебит-кредит эргэлтийн харьцаа", "description": "Дансны дебит/кредит харьцаа. Огцом хазайлт = нэг чиглэлтэй хэвийн бус гүйлгээ.", "isa_ref": "ISA 240 «Залилан» — нэг чиглэлтэй гүйлгээний хэв маяг", "example": "Жишээ: Зөвхөн дебит ихсэх = зардлыг хэтрүүлэн бүртгэх", "icon": "⚖️"},
        "log_turn_d": {"mn_name": "Баримт дебит гүйлгээний хэмжээ", "description": "Нийт дебит гүйлгээний хэмжээ (логарифм масштаб). Том дебит = материаллаг алдааны магадлал өндөр.", "isa_ref": "ISA 320 «Материаллаг байдал»", "example": "Жишээ: 50тэрбум₮-ийн дебит гүйлгээ", "icon": "📊"},
        "log_turn_c": {"mn_name": "Баримт кредит гүйлгээний хэмжээ", "description": "Нийт кредит гүйлгээний хэмжээ (логарифм масштаб).", "isa_ref": "ISA 320 «Материаллаг байдал»", "example": "Жишээ: 30тэрбум₮-ийн кредит бүртгэл", "icon": "📊"},
        "cat_num": {"mn_name": "Дансны ангиллын код", "description": "1xx=Хөрөнгө, 2xx=Өр, 3xx=Өмч, 4-5xx=Орлого, 6-8xx=Зардал, 9xx=Нэгдсэн. Ангилал бүр эрсдэл өөр.", "isa_ref": "ISA 315 «Эрсдэл» — дансны ангиллаар эрсдэл ялгаатай", "example": "Жишээ: 1xx хөрөнгийн дансууд хамгийн олон аномали агуулна", "icon": "🏷️"},
        "log_close_d": {"mn_name": "Жилийн эцсийн дебит үлдэгдэл", "description": "Эцсийн дебит үлдэгдлийн хэмжээ (логарифм масштаб).", "isa_ref": "ISA 505 «Баталгаажуулалт»", "example": "Жишээ: Хөрөнгийн жилийн эцсийн үлдэгдэл", "icon": "📋"},
        "log_close_c": {"mn_name": "Жилийн эцсийн кредит үлдэгдэл", "description": "Эцсийн кредит үлдэгдлийн хэмжээ (логарифм масштаб).", "isa_ref": "ISA 505 «Баталгаажуулалт»", "example": "Жишээ: Өр төлбөрийн жилийн эцсийн үлдэгдэл", "icon": "📋"},
        "year": {"mn_name": "Тайлант жил", "description": "2023/2024/2025. Бүртгэлийн бодлого өөрчлөгдсөн жилд хэвийн бус байдал нэмэгдэж болно.", "isa_ref": "ISA 315 — хугацааны нөлөө", "example": "Жишээ: 2025 онд шинэ стандарт нэвтэрсэн", "icon": "📅"},
    }

    def show_xai_feature_details(self, feature_importances=None):
        if feature_importances is None:
            feature_importances = {"log_abs_change": 0.5769, "log_turn_d": 0.1489, "log_turn_c": 0.1223, "cat_num": 0.0573, "log_close_c": 0.0522, "log_close_d": 0.0213, "turn_ratio": 0.0125, "year": 0.0086}
        st.markdown("### 📖 Шинж чанар бүрийн нарийвчилсан тайлбар")
        for feat_name, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
            info = self.FEATURE_DESCRIPTIONS.get(feat_name, {})
            if not info:
                continue
            pct = importance * 100
            if importance > 0.15:
                bar_color, level = "#c62828", "🔴 Өндөр нөлөөтэй"
            elif importance > 0.05:
                bar_color, level = "#e65100", "🟠 Дунд нөлөөтэй"
            else:
                bar_color, level = "#2e7d32", "🟢 Бага нөлөөтэй"
            st.markdown(f"""
            <div style="background-color: #fafafa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 20px; margin-right: 8px;">{info.get('icon','📊')}</span>
                    <b style="font-size: 15px; color: #333;">{info.get('mn_name', feat_name)}</b>
                    <code style="margin-left: 10px; background: #e8eaf6; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{feat_name}</code>
                    <span style="margin-left: auto; font-weight: bold; color: {bar_color}; font-size: 13px;">{importance:.4f} ({pct:.1f}%) — {level}</span>
                </div>
                <div style="background: #e0e0e0; border-radius: 4px; height: 8px; margin-bottom: 10px;">
                    <div style="background: {bar_color}; width: {min(pct*1.7,100):.0f}%; height: 8px; border-radius: 4px;"></div>
                </div>
                <p style="color: #555; font-size: 13px; margin: 4px 0;">{info.get('description','')}</p>
                <p style="color: #1565C0; font-size: 12px; margin: 4px 0;">📌 <i>{info.get('isa_ref','')}</i></p>
                <p style="color: #777; font-size: 12px; margin: 4px 0 0 0;">{info.get('example','')}</p>
            </div>
            """, unsafe_allow_html=True)

    def show_xai_interpretation(self):
        st.markdown("""
        ---
        <div style="background-color: #F3E5F5; padding: 15px; border-radius: 8px; border-left: 4px solid #7B1FA2;">
            <b>💡 Тайлбарлагдах ХОУ-ын утга учир:</b><br>
            Загвар нь «хар хайрцаг» биш — шийдвэр бүр тайлбартай, ISA стандартуудтай нийцнэ.<br><br>
            ⚠️ Аудитор загварын шийдвэрт бүрэн итгэх бус, <b>мэргэжлийн хүрээнд үнэлэх</b> зарчмыг баримтална (ISA 500).
        </div>
        """, unsafe_allow_html=True)

    def show_list_description(self):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff3e0 0%, #fffaf0 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #e65100; margin-bottom: 20px;">
            <h4 style="color: #bf360c; margin-top: 0;">📋 Хэвийн бус дансуудын жагсаалт</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7;">
                Нэгдсэн санал нэгтгэл (Ensemble)-ээр хэвийн бус гэж тэмдэглэгдсэн
                <b>бүх дансуудын дэлгэрэнгүй</b>. Анхан шатны баримтын шалгалтад шууд ашиглана.
            </p>
            <p style="color: #555; font-size: 13px; margin-bottom: 0;">
                📌 <i>ISA 330 «Аудиторын хариу арга хэмжээ» — эрсдэлтэй дансуудад чиглэсэн шалгалт.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("ℹ️ Баганууд ба шүүлтүүрийн тайлбар", expanded=False):
            st.markdown("""
| Багана | Тайлбар |
|--------|---------|
| **year** | Тайлант жил |
| **account_code** | Дансны код (жишээ: 101-00-01-000) |
| **account_name** | Дансны нэр |
| **turnover_debit / credit** | Нийт дебит / кредит гүйлгээ |
| **turn_ratio** | Дебит/Кредит харьцаа (1.0 = тэнцүү) |
| **log_abs_change** | Дансны цэвэр өөрчлөлт (логарифм) |

**📥 CSV татах** товчоор жагсаалтыг татаж, ажлын баримтад хавсаргана.
            """)

    def show_list_interpretation(self, n_anomalies=0):
        st.markdown(f"""
        ---
        <div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; border-left: 4px solid #E65100;">
            <b>💡 Зөвлөмж:</b><br>
            • Нийт <b>{n_anomalies}</b> хэвийн бус данс илэрсэн<br>
            • turn_ratio өндөр + log_abs_change өндөр дансуудыг <b>нэн тэргүүнд</b> шалгах<br>
            • CSV файлыг татаж аудитын ажлын баримтад хавсаргах
        </div>
        """, unsafe_allow_html=True)

    def show_risk_matrix_description(self):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0f2f1 0%, #f0faf9 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #00695c; margin-bottom: 20px;">
            <h4 style="color: #004d40; margin-top: 0;">🎯 Эрсдэлийн матриц — Харилцагч × Данс хос</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7;">
                Харилцагч тус бүрийн данс, сараар нэгтгэсэн <b>дансны бичилтийн тооуудын шинжилгээ</b>.
                Тодорхой харилцагчтай хэвийн бус хэмжээний, давтамжийн гүйлгээ байгааг илрүүлнэ.
            </p>
            <p style="color: #555; font-size: 13px; margin-bottom: 0;">
                📌 <i>ISA 550 «Холбоотой этгээд» — хэвийн бус гүйлгээ илрүүлэх нь залилангийн эрсдэлийг бууруулна.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("ℹ️ Эрсдэлийн оноо хэрхэн тооцогддог вэ?", expanded=False):
            st.markdown("""
**Дансны бичилтийн тоо** = Харилцагч × Данс × Сар гэсэн гурвалсан хослол.

| Шалгуур | Нөхцөл | Оноо |
|---------|--------|------|
| Гүйлгээний дүн | 75-р хувийн босго (P75) давсан | +1 |
| Гүйлгээний давтамж | 75-р хувийн босго (P75) давсан | +1 |

Оноо 0 = хэвийн, 1 = анхаарах, **2 = нэн тэргүүнд шалгах**.
            """)

    def show_risk_matrix_interpretation(self, n_pairs=0, top_counterparty=""):
        st.markdown(f"""
        ---
        <div style="background-color: #E0F2F1; padding: 15px; border-radius: 8px; border-left: 4px solid #00695C;">
            <b>💡 Зөвлөмж:</b><br>
            • Нийт <b>{n_pairs:,}</b> дансны бичилтийн тоо шинжлэгдсэн<br>
            • Топ 20 харилцагчаас эхлэн анхан шатны баримтыг шалгах<br>
            • Нэг харилцагчтай олон дансаар давтагдсан гүйлгээнд анхаарах
        </div>
        """, unsafe_allow_html=True)

    def show_monthly_trend_description(self):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8eaf6 0%, #f5f5fc 100%);
             padding: 20px; border-radius: 12px; border-left: 5px solid #283593; margin-bottom: 20px;">
            <h4 style="color: #1a237e; margin-top: 0;">📈 Сарын чиг хандлага</h4>
            <p style="color: #333; font-size: 14px; line-height: 1.7;">
                Сар бүрийн гүйлгээний хэмжээ, тооны чиг хандлагыг графикаар харуулна.
                Хэвийн улирлын хэлбэлзэл ба хэвийн бус оргилыг ялгах боломж олгоно.
            </p>
            <p style="color: #555; font-size: 13px; margin-bottom: 0;">
                📌 <i>ISA 520 «Аналитик горим» — хүлээгдэж буй утгатай харьцуулж зөрүүг тодорхойлно.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("ℹ️ Графикуудыг хэрхэн унших вэ?", expanded=False):
            st.markdown("""
| График | Юу хайх вэ? |
|--------|------------|
| **Дээд** (шугаман) — сарын эргэлт | Огцом оргил = тусгай шалгалт |
| **Доод** (баганан) — гүйлгээний тоо | Тоо буурсан ч дүн өссөн = том гүйлгээ |

🟢 12-р сарын оргил = хэвийн (жилийн эцсийн тооцоо)
🔴 Бусад сарын 3+ дахин өсөлт = хэвийн бус → шалгах
            """)

    def show_monthly_trend_interpretation(self):
        st.markdown("""
        ---
        <div style="background-color: #E8EAF6; padding: 15px; border-radius: 8px; border-left: 4px solid #283593;">
            <b>💡 Зөвлөмж:</b><br>
            • Гурван жилийн хэв маягийг харьцуулж огцом өөрчлөлт гарсан сарыг тодорхойлох<br>
            • 12-р сарын оргил = хэвийн (жилийн эцсийн тооцоо)<br>
            • Гүйлгээний тоо буурсан ч дүн өссөн = <b>том дүнтэй цөөн гүйлгээ</b>-д анхаарах
        </div>
        """, unsafe_allow_html=True)

    def show_dashboard_footer(self):
        st.markdown("""
        ---
        <div style="background: linear-gradient(135deg, #263238 0%, #37474f 100%);
             padding: 20px; border-radius: 12px; color: white;">
            <h4 style="color: #80CBC4; margin-top: 0;">🏛️ ISA стандартын нийцлийн тойм</h4>
            <table style="color: #cfd8dc; font-size: 13px; width: 100%%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 8px; width: 30%%;"><b>📊 Нэгтгэл</b></td>
                    <td style="padding: 8px;">ISA 300 — Аудитын стратеги, хүрээ</td></tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 8px;"><b>🔍 Хэвийн бус данс</b></td>
                    <td style="padding: 8px;">ISA 315 — Эрсдэл тодорхойлох; ISA 240 — Залилан</td></tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 8px;"><b>⚖️ ХОУ ↔ Уламжлалт</b></td>
                    <td style="padding: 8px;">ISA 200 — Илрүүлэлтийн эрсдэл бууруулалт</td></tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 8px;"><b>🧠 Тайлбарлагдах ХОУ</b></td>
                    <td style="padding: 8px;">ISA 500 — Аудитын нотолгоо</td></tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 8px;"><b>📋 Дансуудын жагсаалт</b></td>
                    <td style="padding: 8px;">ISA 330 — Эрсдэлд чиглэсэн шалгалт</td></tr>
                <tr style="border-bottom: 1px solid #455a64;">
                    <td style="padding: 8px;"><b>🎯 Эрсдэлийн матриц</b></td>
                    <td style="padding: 8px;">ISA 550 — Холбоотой этгээд</td></tr>
                <tr>
                    <td style="padding: 8px;"><b>📈 Сарын хандлага</b></td>
                    <td style="padding: 8px;">ISA 520 — Аналитик горим</td></tr>
            </table>
            <p style="color: #90a4ae; font-size: 11px; margin-top: 12px; margin-bottom: 0;">
                ⚠️ Загвар нь аудиторыг орлохгүй — аудиторын чадамжийг нэмэгдүүлнэ. Бүх шийдвэрийг мэргэжлийн дүгнэлтээр баталгаажуулна.
            </p>
        </div>
        """, unsafe_allow_html=True)
