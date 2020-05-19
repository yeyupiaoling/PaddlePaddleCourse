package com.yeyupiaoling.lite;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class Utils {

    // 获取预测值中最大概率的标签
    public static int getMaxResult(float[] result) {
        float probability = result[0];
        int r = 0;
        for (int i = 0; i < result.length; i++) {
            if (probability < result[i]) {
                probability = result[i];
                r = i;
            }
        }
        return r;
    }

    // 对将要预测的图片进行预处理
    public static float[] getScaledMatrix(Bitmap image, int desWidth, int desHeight, int channels) {
        float[] dataBuf = new float[channels * desWidth * desHeight];
        int rIndex;
        int gIndex;
        int bIndex;
        int[] pixels = new int[desWidth * desHeight];
        Bitmap bm = Bitmap.createScaledBitmap(image, desWidth, desHeight, false);
        bm.getPixels(pixels, 0, desWidth, 0, 0, desWidth, desHeight);
        int j = 0;
        int k = 0;
        for (int i = 0; i < pixels.length; i++) {
            int clr = pixels[i];
            j = i / desHeight;
            k = i % desWidth;
            rIndex = j * desWidth + k;
            gIndex = rIndex + desHeight * desWidth;
            bIndex = gIndex + desHeight * desWidth;
            // 转成RGB通道顺序，并除以255，跟训练的预处理一样
            dataBuf[bIndex] = (float) (((clr & 0x00ff0000) >> 16) / 255.0);
            dataBuf[gIndex] = (float) (((clr & 0x0000ff00) >> 8) / 255.0);
            dataBuf[rIndex] = (float) (((clr & 0x000000ff)) / 255.0);

        }
        if (bm.isRecycled()) {
            bm.recycle();
        }
        return dataBuf;
    }

    // 压缩图片，避免图片过大
    public static Bitmap getScaleBitmap(String filePath) {
        BitmapFactory.Options opt = new BitmapFactory.Options();
        opt.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(filePath, opt);

        int bmpWidth = opt.outWidth;
        int bmpHeight = opt.outHeight;

        int maxSize = 500;

        // compress picture with inSampleSize
        opt.inSampleSize = 1;
        while (true) {
            if (bmpWidth / opt.inSampleSize < maxSize || bmpHeight / opt.inSampleSize < maxSize) {
                break;
            }
            opt.inSampleSize *= 2;
        }
        opt.inJustDecodeBounds = false;
        return BitmapFactory.decodeFile(filePath, opt);
    }


    // 根据相册返回的URI返回图片的绝对路径
    public static String getPathFromURI(Context context, Uri uri) {
        String result;
        Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
        if (cursor == null) {
            result = uri.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            result = cursor.getString(idx);
            cursor.close();
        }
        return result;
    }


    // 复制莫模型文件到缓存目录
    public static void copyFileFromAsset(Context context, String assets_path, String new_path) {
        File new_file = new File(new_path);
        if (new_file.exists()) {
            return;
        }
        File father_path = new File(new File(new_path).getParent());
        if (!father_path.exists()) {
            father_path.mkdirs();
        }
        try {
            InputStream is = context.getAssets().open(assets_path);
            FileOutputStream fos = new FileOutputStream(new_file);
            byte[] buffer = new byte[1024];
            int byteCount;
            while ((byteCount = is.read(buffer)) != -1) {
                fos.write(buffer, 0, byteCount);
            }
            fos.flush();
            is.close();
            fos.close();
            Log.d("utils", "the model file is copied");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}