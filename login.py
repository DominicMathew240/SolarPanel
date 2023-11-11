import streamlit as st

# Hardcoded username and password (for demonstration purposes)
valid_username = "user"
valid_password = "password"

# Streamlit app
def main():
    st.title("Simple Authentication Example")


    # If the user is authenticated, show the authenticated content
    if "authenticated" in valid_username and "authenticated" in valid_password:
        test_page()
    else:
        login_page()

def login_page():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Check if the entered credentials are valid
    if st.button("Login"):
        if username == valid_username and password == valid_password:
            st.success("Logged In as {}".format(username))
            st.info("Redirecting to the test page...")
            # Redirect to the test page
            test_page()
        else:
            st.warning("Invalid credentials. Please try again.")

def test_page():
    st.title("Welcome to the Test Page!")
    st.write("This is the authenticated content.")
    st.write("You can add more content specific to the test page here.")
    st.markdown('<a href="test.py">AI Model</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
