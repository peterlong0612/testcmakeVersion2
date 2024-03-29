package com.example.myapplication;

import android.content.Intent;
import android.os.Handler;

import android.os.Bundle;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;

public class OpenActivity extends AppCompatActivity {
    private Handler handler;
    private Runnable runnable;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_open);
        //flag使我们必须要设置的变量
        int flag = WindowManager.LayoutParams.FLAG_FULLSCREEN;
        //设置当前窗体为全屏显示
        getWindow().setFlags(flag, flag);
        /**
         * 正常情况下不点击跳过
         */
        handler = new Handler();
        handler.postDelayed(runnable = new Runnable() {
            @Override
            public void run() {
                //跳转到首界面
                Intent intent = new Intent(OpenActivity.this, MainActivity.class);
                startActivity(intent);
                finish();
            }
        }, 2000);//延迟5S后发送handler信息
    }
}
