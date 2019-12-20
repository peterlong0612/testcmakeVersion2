package com.example.myapplication;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.widget.ImageViewCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

//import org.opencv.core.Mat;
//import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

//import static org.opencv.imgcodecs.Imgcodecs.imread;


public class MainActivity extends AppCompatActivity {
    public String TAG="MainActivity";
    //private static final int REQUEST_VIDEO_CAPTURE = 1;

    private ImageView imgViewA;
    private ImageView imgViewB;

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int PICK_IMAGE = 1;
    //private static final int PICK_VIDEO = 2;
    String strdecode="sdcard/rp.png";

    public Uri imgUri;
    private Bitmap mbitmap;
    private Bitmap obitmap;

    private Button mBtn;

    private ImageView iv;//白平衡
    private ImageView lu;//亮度提升
    private ImageView re;//红眼
    private ImageView oof;//失焦
    private ImageView rp;//图像修复

    private Button btn_bph;
    private Button btn_lu;
    private Button btn_rp;
    private Button btn_re;
    private Button btn_main_select;



    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }
    String filename1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initBtn();
        initView();
        //Button button2 = (Button) findViewById(R.id.button2);
        //Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.ic_launcher_background);

        // Example of a call to a native method

        //tv.setText(stringFromJNI());
        //iv = findViewById(R.id.imageView7);
        //re = findViewById(R.id.redeye);
        //lu = findViewById(R.id.lightup);
        //oof = findViewById(R.id.oof);
        //rp = findViewById(R.id.rp);

        //matrixFromJNI(bitmap);
        PermissionUtils.isGrantExternalRW(this,1);

    }

    private void initBtn(){
        btn_bph =  findViewById(R.id.btn_bph);
        btn_lu =  findViewById(R.id.btn_lu);
        btn_rp = findViewById(R.id.btn_rp);
        btn_re = findViewById(R.id.btn_re);
        btn_main_select = findViewById(R.id.btn_main_select);
    }
    private void initView(){
        imgViewA = findViewById(R.id.imgViewA);
        imgViewB = findViewById(R.id.imgViewB);
        imgViewA.setImageDrawable(null);
        imgViewB.setImageDrawable(null);
    }

    /*
    * 打开本地相册选择图片
    * */
    public void chooseImage() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"),
                PICK_IMAGE);
    }

    public native void matrixFromJNI(Object bitmap);//白平衡实现函数
    public native void repair(Object bitmap);
    public native void lightup(Object bitmap);//亮度提高实现函数
    public native void redeye(Object bitmap,String CascadeFileName);


    public void OnClick(View v){
        switch (v.getId()){
            case R.id.btn_main_select:
            {
                chooseImage();
                break;
            }

            case R.id.btn_rp:
            {
                //Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.rp);//drawable.rp为需要的图片
                String strdecode="sdcard/rp.png";
                Bitmap bitmap = BitmapFactory.decodeFile(strdecode); //pathname
                System.out.println("图像修复");

                System.out.println(bitmap);
                bitmap.setHasAlpha(false);
                repair(bitmap);
                bitmap.setHasAlpha(false);
                //bitmap.setConfig(Bitmap.Config.RGB_565);

                strdecode="sdcard/rpoutput.jpg";
                Bitmap bitmap1 = BitmapFactory.decodeFile(strdecode);
                //rp.setImageBitmap(bitmap1);
                imgViewB.setImageBitmap(bitmap1);
                break;
            }
            case R.id.btn_bph:
            {
               // TextView tv = findViewById(R.id.sample_text);
                //tv.setText("changed");//查看图片是否改变，用于初期实验用，可删除
                //Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.pic1);//drawable.pic1为白平衡需要的图片
                System.out.println("白平衡");
                System.out.println(obitmap);
                matrixFromJNI(obitmap);
                //iv.setImageBitmap(bitmap);
                imgViewB.setImageBitmap(obitmap);
                break;
            }
            case R.id.btn_lu:
            {
                //Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.lu);//drawable.lu为需要的图片

                System.out.println("亮度提升");

                System.out.println(obitmap);
                lightup(obitmap);
                //lu.setImageBitmap(bitmap);
                imgViewB.setImageBitmap(obitmap);
                break;
            }
            case R.id.btn_re:
            {
                //Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.redeye);//drawable.lu为需要的图片
                Bitmap bitmap = BitmapFactory.decodeFile("sdcard/redeyepic.jpg");
                System.out.println("去红眼");

                System.out.println(bitmap);
                /*try{
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_eye);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    File mCascadeFile = new File(cascadeDir, "raw/haarcascade_eye.xml");
                    FileOutputStream os = new FileOutputStream(mCascadeFile);


                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        os.write(buffer, 0, bytesRead);
                        System.out.println("check");

                    }
                    is.close();
                    os.close();*/
                    String s = "sdcard/haarcascade_eye.xml";
                    redeye(bitmap,s);
                    Bitmap bitmap1 = BitmapFactory.decodeFile("sdcard/output.jpg");
                    //re.setImageBitmap(bitmap1);
                    imgViewB.setImageBitmap(bitmap1);
                    /*cascadeDir.delete();
                } catch (IOException e){
                    e.printStackTrace();
                    System.out.println( "Failed to load cascade. Exception thrown: " + e);
                }
                break;*/
            }

        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, final Intent data) {
        //用户操作完成，结果代码返回是-1，即RERULT_OK
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            //todo 获取文件定位，放置选择的图片
            imgUri= data.getData();
            Log.e("imgUri", imgUri.toString());

            ContentResolver cr= this.getContentResolver();
            try{
              //获取图片
                mbitmap = BitmapFactory.decodeStream(cr.openInputStream(imgUri));
                obitmap = BitmapFactory.decodeStream(cr.openInputStream(imgUri));
                imgViewA.setImageBitmap(mbitmap);
            } catch (FileNotFoundException e){
                Log.e("Exception",e.getMessage(),e);
            }
            //imgViewA.setImageURI(imgUri);
            //videoView.start();  //播放视频
        }
        else{
            //操作错误或者没有选择图片
            Log.i("MainActivity","Operation Error!");

        }
        Log.d(TAG, "onActivityResult() called with: requestCode = [" + requestCode + "], resultCode = [" + resultCode + "], data = [" + data + "]");
    }

    /*public native void outoffocus(Object bitmap);
    public void  onclickoof(View view)
    {
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.oof);//drawable.oof为需要的图片
        System.out.println("失焦恢复");
        System.out.println(bitmap);
        outoffocus(bitmap);
        //oof.setImageMatrix(bitmap);

        oof.setImageBitmap(bitmap);
    }*/


    //public native void repair(Object bitmap);
    public void  onclickrp(View view)
    {

        //Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.rp);//drawable.rp为需要的图片
        String strdecode="sdcard/rp.png";
        Bitmap bitmap = BitmapFactory.decodeFile(strdecode); //pathname
        System.out.println("图像修复");

        System.out.println(bitmap);
        bitmap.setHasAlpha(false);
        repair(bitmap);
        bitmap.setHasAlpha(false);
        //bitmap.setConfig(Bitmap.Config.RGB_565);

        strdecode="sdcard/rpoutput.jpg";
        Bitmap bitmap1 = BitmapFactory.decodeFile(strdecode);
        rp.setImageBitmap(bitmap1);
    }

    //public native void lightup(Object bitmap);//亮度提高实现函数
    public void onclicklightup(View view){//触发器
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.lu);//drawable.lu为需要的图片

        System.out.println("亮度提升");

        System.out.println(bitmap);
        lightup(bitmap);
        lu.setImageBitmap(bitmap);
    }

    //public native void redeye(Object bitmap,String CascadeFileName);
    public void onclickredeye(View view){//触发器
        //Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.redeye);//drawable.lu为需要的图片
        Bitmap bitmap = BitmapFactory.decodeFile("sdcard/redeyepic.jpg");
        System.out.println("去红眼");

        System.out.println(bitmap);
        try{
            InputStream is = getResources().openRawResource(R.raw.haarcascade_eye);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "raw/haarcascade_eye.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
                System.out.println("check");

            }
            is.close();
            os.close();

            redeye(bitmap,mCascadeFile.getAbsolutePath());
            Bitmap bitmap1 = BitmapFactory.decodeFile("sdcard/output.jpg");
            re.setImageBitmap(bitmap1);

            cascadeDir.delete();
        } catch (IOException e){
            e.printStackTrace();
            System.out.println( "Failed to load cascade. Exception thrown: " + e);
        }

    }
    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case REQUEST_IMAGE_CAPTURE: {
                //todo 判断权限是否已经授予

                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
                for (int i = 0; i < permissions.length ; i++ ){
                    Log.i("MainActivity","申请的权限为：" + permissions[i] +"，申请结果：" +
                            grantResults[i]);
                }
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED
                        && grantResults[1] == PackageManager.PERMISSION_GRANTED
                        && grantResults[2] == PackageManager.PERMISSION_GRANTED) {
                    //takeVideo();
                    break;
                }


            }
        }
    }

}
