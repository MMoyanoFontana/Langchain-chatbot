import gradio as gr
custom_css = """
:root{
  --c50:#eff6ff;   --c100:#dbeafe; --c200:#bfdbfe; --c300:#93c5fd;
  --c400:#60a5fa; --c500:#3b82f6; --c600:#1861fc; --c700:#1d4ed8;
  --c800:#1e40af; --c900:#1e3a8a; --c950:#16172C;

  --ink:#0a0d1c;
  --bg-soft:#eef2f9;         
  --bg-soft-2:#e7edf8;
  
  --card-bg: #1a1b2e;
  --btn-primary: #1861fc;
}

html, body { height:100%; margin:0; }
body{
  background-color: var(--bg-soft);

}
@media (prefers-color-scheme: dark){
  body{
    background-color:#0c1324;
    background:
      radial-gradient(1200px 800px at 15% 20%, #0b1220 0%, #0e1629 55%, #0a1222 100%);
  }
}

/* Grid centrado y responsivo */
#login-wrapper{
  min-height:100vh;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  padding: 48px;
  gap:32px;
}

/* Login card mejorado */
#login-card{
  --card-w: 520px;
  background: var(--card-bg);
  backdrop-filter: blur(16px);
  color:#e8edf5;
  width:min(96vw, var(--card-w));
  padding: 40px 44px 40px;
  border-radius:24px;
  border:1px solid rgba(255,255,255,.06);
  box-shadow: 
    0 32px 64px -24px rgba(2,6,23,.5),
    0 0 0 1px rgba(255,255,255,.03);
  display:flex; 
  flex-direction:column; 
  gap:28px;
  position:relative;
  animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.2s both;
}

@keyframes fadeInUp{
  from{ opacity:0; transform:translateY(30px); }
  to{ opacity:1; transform:translateY(0); }
}

#login-logo-container{
  text-align:center;
  animation: logoAppear 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

#login-logo{
  width: 320px;
  max-width:90vw;
  margin: 0 auto;
  display: block;
  filter: drop-shadow(0 12px 32px rgba(24,97,252,0.35));
  animation: logoFloat 3s ease-in-out infinite;
}

@keyframes logoAppear{
  from{ 
    opacity:0; 
    transform:translateY(-30px) scale(0.9);
  }
  to{ 
    opacity:1; 
    transform:translateY(0) scale(1);
  }
}



/* Título y textos */
#login-card h1{ 
  margin:0 0 8px; 
  font-size: clamp(26px, 3vw, 32px); 
  line-height:1.15; 
  color:#f8fafc;
  text-align:center;
  font-weight:700;
  letter-spacing:-0.5px;
}

#login-card .subtitle{
  color:#f8fafc;
  text-align:center;
  font-size:15px;
  margin-bottom:12px;
  line-height:1.5;
}

#login-card p,#login-card .gr-markdown{ 
  color:#cbd5e1; 
  font-size:14px;
  font-weight:800;
  margin-bottom:8px;
  letter-spacing:0.2px;
}

/* Inputs mejorados */
#login-card input[type="text"],
#login-card input[type="password"],
#login-username input,
#login-password input{
  width:100%;
  padding: 14px 16px;
  border:1px solid #2a3448; 
  background:#0f1624; 
  color:#e8edf5;
  font-size:15px;
  line-height:1.4;
  outline:none;
  transition: all .25s cubic-bezier(0.16, 1, 0.3, 1);
}

/* Botón hero mejorado */
.gr-button.gr-button--primary,
#login-btn{
  background: var(--bg-soft) !important;
  border:none !important;
  color:#1861fc !important;
  border-radius:14px;
  padding:16px 24px;
  font-weight:700;
  font-size:16px;
  letter-spacing:0.5px;
  text-transform:none;
  box-shadow: 
    0 12px 28px -8px rgba(24,97,252,0.5),
    0 0 0 1px rgba(255,255,255,0.15) inset,
    0 1px 2px 0 rgba(255,255,255,0.25) inset;
  transition: all .25s cubic-bezier(0.16, 1, 0.3, 1);
  margin-top:12px;
  cursor:pointer;
  position:relative;
  overflow:hidden;
}

.gr-button.gr-button--primary::before,
#login-btn::before{
  content:'';
  position:absolute;
  top:0;
  left:-100%;
  width:100%;
  height:100%;
  background:linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  transition:left 0.5s ease;
}

.gr-button.gr-button--primary:hover::before,
#login-btn:hover::before{
  left:100%;
}

.gr-button.gr-button--primary:hover,
#login-btn:hover{ 
  transform: translateY(-3px) scale(1.01); 
  box-shadow: 
    0 20px 40px -12px rgba(24,97,252,0.6),
    0 0 0 1px rgba(255,255,255,0.2) inset,
    0 1px 2px 0 rgba(255,255,255,0.3) inset;
  filter:brightness(1.1);
} 

.gr-button.gr-button--primary:active,
#login-btn:active{ 
  transform: translateY(-1px) scale(0.99); 
  box-shadow: 
    0 8px 20px -8px rgba(24,97,252,0.5),
    0 0 0 1px rgba(255,255,255,0.15) inset;
}

#login-btn[disabled]{ 
  opacity:.4; 
  cursor:not-allowed; 
  box-shadow:none;
  transform:none !important;
  filter:grayscale(1);
}


/* Mensajes mejorados */
#login-message{ 
  font-size:14px; 
  margin-top:4px; 
  padding:12px 16px;
  border-radius:12px;
  text-align:center;
  font-weight:600;
  opacity:0;
}


#login-message.success{ 
  color:#10b981; 
  background:rgba(16,185,129,0.12);
  border:1.5px solid rgba(16,185,129,0.25);
  box-shadow:0 4px 12px -4px rgba(16,185,129,0.3);
}

#login-message{
  color:var(--bg-soft);
  animation: shake .32s ease-in-out 1;
}
/* Si el navegador soporta :has, sacude toda la card e indica error en inputs */
#login-card:has(#login-message.error){ animation: shake .32s ease-in-out 1; }

/* Shake */
@keyframes shake{
  0%{ transform:translateX(0) } 20%{ transform:translateX(-6px) }
  40%{ transform:translateX(6px) } 60%{ transform:translateX(-4px) }
  80%{ transform:translateX(4px) } 100%{ transform:translateX(0) }
}

"""

gemis_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eff6ff",
        c100="#dbeafe",
        c200="#bfdbfe",
        c300="#93c5fd",
        c400="#60a5fa",
        c500="#3b82f6",
        c600="#1861fc",
        c700="#1d4ed8",
        c800="#1e40af",
        c900="#1e3a8a",
        c950="#16172C",
    ),
    text_size="md",
    radius_size="md",
).set()
