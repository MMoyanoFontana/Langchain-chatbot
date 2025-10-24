from chat_interface import create_chat_interface
from auth import _auth

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(auth=_auth, auth_message="Ingresa tus credenciales para continuar.", show_error=True, share=True)
