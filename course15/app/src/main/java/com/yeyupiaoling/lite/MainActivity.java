package com.yeyupiaoling.lite;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.Tensor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private String model_path;
    // 模型文件
    String modelPath = "mobilenet_v1.nb";
    private boolean load_result = false;
    // 输入图片的形状，分别是：batch size、通道数、宽度、高度
    private long[] ddims = {1, 3, 224, 224};
    private ImageView imageView;
    private TextView showTv;
    private PaddlePredictor paddlePredictor;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        model_path = getCacheDir().getAbsolutePath() + File.separator + modelPath;
        // 初始化控件
        initView();
        // 动态请求权限
        requestPermissions();
        // 从assets中复制模型文件到缓存目录下
        Utils.copyFileFromAsset(this, modelPath, model_path);
    }

    // 初始化控件
    private void initView(){
        Button loadBtn = findViewById(R.id.load);
        Button clearBtn = findViewById(R.id.clear);
        Button inferBtn = findViewById(R.id.infer);
        showTv = findViewById(R.id.show);
        imageView = findViewById(R.id.image_view);



        // 加载模型点击事件
        loadBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                MobileConfig config = new MobileConfig();
                config.setModelFromFile(model_path);
                config.setThreads(2);
                paddlePredictor = PaddlePredictor.createPaddlePredictor(config);

                if (paddlePredictor != null) {
                    load_result = true;
                    Toast.makeText(MainActivity.this, "模型加载成功", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(MainActivity.this, "模型加载失败", Toast.LENGTH_SHORT).show();
                }
            }
        });

        // 清空模型点击事件
        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                paddlePredictor = null;
                load_result = false;
                Toast.makeText(MainActivity.this, "模型已清空", Toast.LENGTH_SHORT).show();
            }
        });

        // 打开相册选择图片预测点击事件
        inferBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (load_result){
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, 1);
                } else {
                    Toast.makeText(MainActivity.this, "模型未加载", Toast.LENGTH_SHORT).show();
                }
            }
        });

    }


    // 回调事件
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        String image_path;
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 1) {
                if (data == null) {
                    return;
                }
                // 获取相册返回的URI
                Uri image_uri = data.getData();
                // 根据图片的URI获取绝对路径
                image_path = Utils.getPathFromURI(MainActivity.this, image_uri);
                // 压缩图片用于显示
                Bitmap bitmap = Utils.getScaleBitmap(image_path);
                imageView.setImageBitmap(bitmap);
                // 开始预测图片
                predictImage(image_path);
            }
        }
    }


    // 根据图片的路径预测图片
    private void predictImage(String image_path) {
        // 把图片进行压缩
        Bitmap bmp = Utils.getScaleBitmap(image_path);
        // 把图片转换成浮点数组，用于预测
        float[] inputData = Utils.getScaledMatrix(bmp, (int)ddims[2], (int)ddims[3], (int)ddims[1]);
        try {
            long start = System.currentTimeMillis();
            // 执行预测，获取预测结果
            Tensor input = paddlePredictor.getInput(0);
            input.resize(ddims);
            input.setData(inputData);
            paddlePredictor.run();
            float[] result = paddlePredictor.getOutput(0).getFloatData();
            long end = System.currentTimeMillis();
            // 获取概率最大的标签
            int r = Utils.getMaxResult(result);
            // 获取标签对应的类别名称
            String[] names = {"苹果", "哈密瓜", "樱桃", "葡萄", "梨", "西瓜"};
            String show_text = "标签：" + r + "\n名称：" + names[r] + "\n概率：" + result[r] + "\n时间：" + (end - start) + "ms";
            // 显示预测结果
            showTv.setText(show_text);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 多权限动态申请
    private void requestPermissions() {
        List<String> permissionList = new ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        }

        if (!permissionList.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissionList.toArray(new String[permissionList.size()]), 1);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
            if (grantResults.length > 0) {
                for (int i = 0; i < grantResults.length; i++) {

                    int grantResult = grantResults[i];
                    if (grantResult == PackageManager.PERMISSION_DENIED) {
                        String s = permissions[i];
                        Toast.makeText(this, s + " permission was denied", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        }
    }
}
