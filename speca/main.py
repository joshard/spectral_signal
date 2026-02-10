from kivy import platform

# ----------------------------------------------------------
# 1. REQUEST ANDROID PERMISSIONS 
# ----------------------------------------------------------
if platform == "android":
    from android.permissions import request_permissions, Permission
    request_permissions([
        Permission.CAMERA,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE
    ])

# ----------------------------------------------------------
# 2. Import lightweight modules
# ----------------------------------------------------------
from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image

# ----------------------------------------------------------
# 3. Import heavy modules
# ----------------------------------------------------------
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.toast import toast

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA
from catboost import CatBoostClassifier


class SpectralCamera(MDApp):

    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Camera
        self.cameraObject = Camera(play=True)

        # Image to show spectrum graph
        self.Image = Image(source="memo_data/spec_graph_Leaf.png")

        # Prediction label
        self.prediction_label = MDLabel(
            text="Prediction: [No prediction yet]",
            halign="center",
            theme_text_color="Primary",
            font_style="H6",
            size_hint_y=None,
            height=50
        )

        # Capture button
        self.camaraClick = Button(text="Take Photo",
                                  size_hint=(.5, .2),
                                  pos_hint={'x': .25, 'y': .75})

        self.camaraClick.bind(on_press=self.onCameraClick)
        self.camaraClick.bind(on_release=self.reload_image)

        # Add widgets
        layout.add_widget(self.cameraObject)
        layout.add_widget(self.Image)
        layout.add_widget(self.prediction_label)
        layout.add_widget(self.camaraClick)

        return layout

    # Capture photo
    def onCameraClick(self, *args):
        self.cameraObject.export_to_png('memo_data/spec_sample.png')
        toast("Captured")

    # Run ML model and reload graph
    def reload_image(self, *args):
        data = self.spec_image('memo_data/spec_sample.png')

        model = CatBoostClassifier()
        model.load_model("model.cbm")
        preds = model.predict(data)

        if len(preds) > 0:
            pred_text = str(preds[0]).strip("[]")
            self.prediction_label.text = f"Prediction: {pred_text}"
        else:
            self.prediction_label.text = "Prediction: [No prediction available]"

        self.spec_graph(data)
        self.Image.reload()

    # Extract spectral features
    def spec_image(self, path):
        img = cv2.imread(path)
        q = np.random.randint(img.shape[2])
        band = img[:, :, q]
        df = pd.DataFrame(band)
        df2 = df.mean(axis=1)

        transformer = KernelPCA(n_components=30, kernel='linear')
        Xt = transformer.fit_transform(pd.DataFrame(df2))

        df = pd.DataFrame(Xt)
        return df.mean(axis=1)

    # Plot and save graph
    def spec_graph(self, data):
        plt.clf()
        plt.plot(data)
        plt.xlabel("Wavelength")
        plt.ylabel("Absorption")
        plt.savefig('memo_data/spec_graph_Leaf.png')
        plt.clf()
        data.to_csv('data/file1.csv')


if __name__ == '__main__':
    SpectralCamera().run()
