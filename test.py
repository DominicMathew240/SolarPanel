import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager

def run(img_dir, labels):
    st.set_option("deprecation.showfileUploaderEncoding", False)
    idm = ImageDirManager(img_dir)

    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    else:
        idm.set_all_files(st.session_state["files"])
        idm.set_annotation_files(st.session_state["annotation_files"])
    
    def refresh():
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0

    def next_image():
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning('This is the last image.')

    def previous_image():
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning('This is the first image.')

    def next_annotate_file():
        image_index = st.session_state["image_index"]
        next_image_index = idm.get_next_annotation_image(image_index)
        if next_image_index:
            st.session_state["image_index"] = idm.get_next_annotation_image(image_index)
        # else:
        #     st.warning("All images are annotated.")
        #     next_image()

    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index

    # Sidebar: show status
    n_files = len(st.session_state["files"])
    n_annotate_files = len(st.session_state["annotation_files"])
    st.sidebar.write("Total files:", n_files)
    st.sidebar.write("Total annotate files:", n_annotate_files)
    st.sidebar.write("Remaining files:", n_files - n_annotate_files)

    # Add file uploader to the sidebar and allow multiple files upload
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_file is not None:
        for uploaded_file in uploaded_file:
            file_name = uploaded_file.name
            if file_name not in st.session_state["files"]:
                st.session_state["files"].append(file_name)
                st.session_state["image_index"] = len(st.session_state["files"]) - 1
                with open(os.path.join(img_dir, file_name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
        # else:
        #     st.warning("Upload a File...")

    new_label = st.sidebar.text_input("Enter your new label")

    if st.sidebar.button("Add Label"):
        if new_label:
            # Add the new label to the labels list
            labels.append(new_label)
            st.success(f"Label {new_label} added.")
        else:
            st.warning("Please enter a label name.")

    st.sidebar.selectbox(
        "Files",
        st.session_state["files"],
        index=st.session_state["image_index"],
        on_change=go_to_image,
        key="file",
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Previous image", on_click=previous_image)
    with col2:
        st.button(label="Next image", on_click=next_image)
    st.sidebar.button(label="Next need annotate", on_click=next_annotate_file)
    st.sidebar.button(label="Refresh", on_click=refresh)

    # Main content: annotate images
    rects = None
    if st.session_state["image_index"] < len(st.session_state["files"]):
        img_file_name = idm.get_image(st.session_state["image_index"])
        img_path = os.path.join(img_dir, img_file_name)
        im = ImageManager(img_path)
        img = im.get_img()
        resized_img = im.resizing_img()
        resized_rects = im.get_resized_rects()
        rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

        # Display the image
        # st.image(img, caption=img_file_name)
    else:
        st.warning("Add an images to annotate.")
    def annotate():
        im.save_annotation()
        image_annotate_file_name = img_file_name.split(".")[0] + ".xml"
        if image_annotate_file_name not in st.session_state["annotation_files"]:
            st.session_state["annotation_files"].append(image_annotate_file_name)
        next_annotate_file()

    # If user annotate the image out of the box create a popup to reposition the box
    try: 
        if rects:     
            st.button(label="Save", on_click=annotate)
            preview_imgs = im.init_annotation(rects)

            for i, prev_img in enumerate(preview_imgs):
                prev_img[0].thumbnail((200, 200))
                col1, col2 = st.columns(2)
                with col1:
                    col1.image(prev_img[0])
                with col2:
                    default_index = 0
                    if prev_img[1]:
                        default_index = labels.index(prev_img[1])

                    select_label = col2.selectbox(
                        "Label", labels, key=f"label_{i}", index=default_index
                    )
                    im.set_annotation(i, select_label)
    except:
        st.warning("Please annotate the image inside the box.")
            

if __name__ == "__main__":
    custom_labels = [""]
    run("img_dir", custom_labels)