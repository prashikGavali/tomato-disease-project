package com.example.tomato_disease_detect;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.tomato_disease_detect.ml.TomatoesApp;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 128;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        // Set the locale of the application to Marathi
        Locale locale = new Locale("mr");
        Locale.setDefault(locale);
        Configuration config = new Configuration();
        config.locale = locale;
        getBaseContext().getResources().updateConfiguration(config, getBaseContext().getResources().getDisplayMetrics());

        camera.setOnClickListener(new View.OnClickListener() {
            // come here if permission related issue arises
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    @SuppressLint("SetTextI18n")
    public void classifyImage(Bitmap image){
        try {
            TomatoesApp model = TomatoesApp.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);

            // Resize and normalize image
            Bitmap resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            int[] intValues = new int[imageSize * imageSize];
            resizedImage.getPixels(intValues, 0, resizedImage.getWidth(), 0, 0, resizedImage.getWidth(), resizedImage.getHeight());
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++){
                    int val = intValues[i * imageSize + j];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                    byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                    byteBuffer.putFloat((val & 0xFF) / 255.0f);
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            TomatoesApp.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

//            Tomato Bacterial Spot = बॅक्टीरियल स्पॉट
//            Tomato Late Blight = उशीर बुरशी
//            Tomato Spider Mite = माशीचा रोग
//            Tomato Yellow Leaf Curl = सुडभुई व्हायरस रोग
//            Tomato Healthy = निरोगी रोप

            HashMap<String, String> remedies = new HashMap<>();
            remedies.put("बॅक्टीरियल स्पॉट", "प्रथम टोकांनंतर फवारणी करण्यापूर्वी टोमॅटोच्या पानांवरील सर्व धुळे आरंभीच्या अवस्थेत हटवावे. फवारणी करण्यापूर्वी पूर्णतेच असांबळ नसलेल्या व अधिक संख्येत कीटकांची नाश करण्याचे उपाय करावे. रोगग्रस्त झाडे ताबडतोब न करता, त्यांची काढणी करून नष्ट करावी.\n" + "नोंद: येथील सल्ला केवळ सूचना देण्यासाठी आहे. प्रथम टोकांनंतर किंवा फवारणी करण्यापूर्वी वैद्यकीय सल्ल्याचा वापर करावा.");
            remedies.put("उशीर बुरशी", "बोर्डो मिश्रण, कॉपर ऑक्सीक्लोराईड आणि कॉपर हायड्रोक्साईड फवारणी करा. झाडांना अनियमित अंतरालाने पाणी द्या आणि पाण्याच्या स्तरावर नियंत्रण ठेवा. वाढण्याची नंतरची फवारणी कमी करा आणि सध्याच्या फवारणीत फवारणी करा.\n" + "नोंद: येथील सल्ला केवळ सूचना देण्यासाठी आहे. फवारणी करण्यापूर्वी वैद्यकीय सल्ल्याचा वापर करावा.");
            remedies.put("माशीचा रोग", "टमाटर झाडांवरील किड नियंत्रण करण्यासाठी द्राक्षाचे रस आणि सोडा शोषणी घ्या. टमाटर झाडांच्या पानांवरील किड आणि फारसे दूर करण्यासाठी लस सुद्धा एक मार्ग आहे.\n" + "वरील उपाय सूची फक्त सूचनात्मक आहे. त्यामुळे आपल्या क्षेत्रातील स्थानिक विशेषज्ञांच्या सल्ल्यासाठी संपर्क करणे आवश्यक आहे.");
            remedies.put("सुडभुई व्हायरस रोग", "टोमॅटो पिकांसाठी उचित वातावरण तयार करणे, आरोग्यपूर्ण टोमॅटो पिके राखणे आणि उच्च गुणवत्तेचे बियाणे वापरणे रोगाचे फैलवणी थांबवण्यास महत्वाचे आहे. किडीनाशक, जैविक किटकनाशक जसे कि सॅलिवेटे किंवा लसकडी यांचा रोग फैलवणी कमी करण्यासाठी वापर केला जाऊ शकतो.\n" + "नोंद: येथील सल्ला केवळ सूचना देण्यासाठी आहे. फवारणी करण्यापूर्वी वैद्यकीय सल्ल्याचा वापर करावा.");
            remedies.put("निरोगी रोप", "कोणतेही उपाय आवश्यक नाहीत. आपले टोमॅटो रोगमुक्त आहे!\n" + "तरीही अधिक माहितीसाठी डॉक्टरांशी संपर्क साधा.");

            String[] classes = {"बॅक्टीरियल स्पॉट", "उशीर बुरशी", "माशीचा रोग", "सुडभुई व्हायरस रोग", "निरोगी रोप"};
            result.setText(classes[maxPos]);
            String classifiedRemedies = remedies.get(classes[maxPos]);

            TextView confidenceTextView = findViewById(R.id.confidenceTextView);
            float maxConfidence_ = confidences[maxPos];
            String confidenceString = String.format(String.valueOf(maxConfidence_));
            confidenceTextView.setText(confidenceString);

            TextView remediesTextView = findViewById(R.id.remediesTextView);
            remediesTextView.setText(classifiedRemedies);
            remediesTextView.setVisibility(View.VISIBLE);
            remediesTextView.setTextLocale(new Locale("mr"));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
