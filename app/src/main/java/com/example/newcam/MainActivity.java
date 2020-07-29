package com.example.newcam;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private int CAMERA_PERMISSION_CODE = 1;
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    Scalar whiteColor = new Scalar(255, 255, 255);
    boolean Start;
    boolean boardDetected;

    Mat boardMask = null;
    Mat boardContentMask = null;
    Mat board = null;
    Mat hierarchy = null;
    Mat boardHierarchy = null;
    Mat frame = null;
    Mat tempFrame = null;
    Mat originalFrame = null;
    Mat imageROI = null;
    Mat boardROI = null;


    Point[] points = null;

    List<MatOfPoint> LargeContourArray = null;
    List<MatOfPoint> contours = null;
    List<MatOfPoint> boardContours = null;

    Size gaussianKernel;
    Size roiGaussianKernel;

    Rect rectangle;
    Rect roi = null;
    Scalar greenColor;
    Scalar redColor;
    Scalar blueColor;

    MatOfPoint2f approx = null;
    double minWhiteBoardArea = 100000;

    int maxNoiseArea;
    int minHandArea;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(MainActivity.this, "You have already granted the permission to Camera", Toast.LENGTH_SHORT).show();
        } else {
            requestCameraPermission();
        }

        setContentView(R.layout.activity_main);


        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setMaxFrameSize(1280, 720);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch (status) {
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }

    private void cleanVariables() {
        frame = null;

        contours = null;
        hierarchy = null;

        boardContours = null;
        boardHierarchy = null;

        boardContentMask = null;
        approx = null;
        imageROI = null;
        roi = null;
        boardROI = null;

        points = null;
        originalFrame = null;
        tempFrame = null;

        System.gc();
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        cleanVariables();


        tempFrame = inputFrame.rgba();
        frame = new Mat();
        Imgproc.cvtColor(tempFrame, frame, Imgproc.COLOR_RGBA2GRAY);
        originalFrame = frame.clone();

        //apply gaussian blur
        Imgproc.GaussianBlur(frame, frame, gaussianKernel, 0);

        //apply Canny's edge detection
        Imgproc.Canny(frame, frame, 100, 200);

        //Find contours
        contours = new ArrayList<>();
        hierarchy = new Mat();
        Imgproc.findContours(frame, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        //frame drawing contours
        //Find largest contour which may be the white board
        double largeContourArea = 0;
        int largeContourIndex = 0;

        if (contours.size() > 0) {
            //finding largest contour
            for (int index = 0; index < contours.size(); index++) {
                double contourArea = Imgproc.contourArea(contours.get(index));
                if (contourArea > largeContourArea) {
                    largeContourArea = contourArea;
                    largeContourIndex = index;
                }
            }
            //if largest contour has the minimum required area

            if (largeContourArea > minWhiteBoardArea) {
                //Draw largest contour
                //Imgproc.drawContours(boardMask, contours, largeContourIndex, new Scalar(255), 3);

                //finding approx polygon for the board
                MatOfPoint2f largeContour = new MatOfPoint2f(contours.get(largeContourIndex).toArray());
                double epsilon = Imgproc.arcLength(largeContour, true);
                approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(largeContour, approx, 0.1 * epsilon, true);
                Point[] points = approx.toArray();

                if (points.length == 4) {
                    MatOfPoint reqContour = new MatOfPoint(points);
                    rectangle = null;
                    rectangle = Imgproc.boundingRect(reqContour);
                    LargeContourArray = new ArrayList<>();
                    LargeContourArray.add(reqContour);
                    Imgproc.drawContours(tempFrame, LargeContourArray, 0, redColor, 3);
                    Imgproc.rectangle(tempFrame, rectangle.tl(), rectangle.br(), greenColor, 3);
                    boardMask = null;
                    boardMask = Mat.zeros(frame.size(), CvType.CV_8UC1);
                    Imgproc.drawContours(boardMask, LargeContourArray, 0, new Scalar(255), -1);
                    boardDetected = true;
                }

            }
        }

        if (Start) {
            if (boardDetected) {
                Core.bitwise_and(originalFrame, boardMask, originalFrame);
                roi = new Rect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
                imageROI = originalFrame.submat(roi);

                //apply blur on the roi
                Imgproc.GaussianBlur(imageROI, imageROI, roiGaussianKernel, 0);
                Imgproc.threshold(imageROI, imageROI, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_TRIANGLE);

                //find text contours
                boardContours = new ArrayList<>();
                boardHierarchy = new Mat();
                boardContentMask = null;
                boardContentMask = Mat.zeros(imageROI.size(), CvType.CV_8UC1);
                Imgproc.findContours(imageROI, boardContours, boardHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                board = Mat.ones(frame.size(), CvType.CV_8UC3).setTo(whiteColor);

                boardROI = board.submat(new Rect((board.cols() - rectangle.width) / 2, (board.rows() - rectangle.height) / 2, rectangle.width, rectangle.height));
                //Imgproc.drawContours(boardROI, boardContours, -1, blueColor , -1);
                for (int index = 0; index < boardContours.size(); index++) {
                    double contourArea = Imgproc.contourArea(boardContours.get(index));
                    if ((contourArea > maxNoiseArea) && (contourArea < minHandArea)) {
                        Imgproc.drawContours(boardROI, boardContours, index, blueColor, -1);
                    }
                }


                //Find required center position of image
                //board.submat(new Rect(0,0,rectangle.width,rectangle.height)).copyTo(imageROI);
                //imageROI.copyTo(board.submat(new Rect(0,0,rectangle.width,rectangle.height)));


                return board;

            }
        }

        return tempFrame;

//        if (Start) {
//
//            if (contours.size() > 0) {
//
//
//
//                if (largeContourArea > minWhiteBoardArea) {
//
//
//                    //If the polygon is a rectangle
//                    if (points.length == 4) {
//                        MatOfPoint reqContour = new MatOfPoint(points);
//                        Rect rectangle = Imgproc.boundingRect(reqContour);
//                        LargeContourArray = new ArrayList<>();
//                        LargeContourArray.add(reqContour);
//                        Imgproc.drawContours(boardMask, LargeContourArray, 0, new Scalar(255) , -1);
//                        //Imgproc.rectangle(boardMask,rectangle.tl(),rectangle.br(),new Scalar(255),-1);
//
//
//                        //Selecting the ROI
//                        Core.bitwise_and(originalFrame,boardMask,boardMask);
//                        roi = new Rect(rectangle.x,rectangle.y,rectangle.width,rectangle.height);
//                        imageROI = originalFrame.submat(roi);
//
//                        //apply blur on the roi
//                        Imgproc.GaussianBlur(imageROI, imageROI, roiGaussianKernel,0);
//                        Imgproc.threshold(imageROI,imageROI,0,255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_TRIANGLE);
//
//                        //Find required center position of image
//                        //board.submat(new Rect(0,0,rectangle.width,rectangle.height)).copyTo(imageROI);
//                        imageROI.copyTo(board.submat(new Rect((board.cols()-rectangle.width)/2,(board.rows()-rectangle.height)/2,rectangle.width,rectangle.height)));
//
//
//
//                        return board;
//
//
//                    }
//
//
//                } else {
//                    Imgproc.drawContours(boardMask, contours, -1, new Scalar(255), 3);
//                    return boardMask;
//                }
//
//            }
//
//
//
//        }

    }

    private void requestCameraPermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
            new AlertDialog.Builder(this)
                    .setTitle("Permission required")
                    .setMessage("Permission to access Camera is needed as the application itself is based on camera")
                    .setPositiveButton("ok", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
                        }
                    })
                    .setNegativeButton("cancel", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            dialog.dismiss();
                        }
                    })
                    .create().show();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);

        }


    }

    public void initiateProcess(View Button) {
        Start = !Start;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        //super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera Permission Granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Start = false;
        minHandArea = 2000;
        maxNoiseArea = 20;
        gaussianKernel = new Size(3, 3);
        boardDetected = false;
        greenColor = new Scalar(0, 255, 0);
        redColor = new Scalar(255, 0, 0);
        blueColor = new Scalar(0, 0, 255);
        roiGaussianKernel = new Size(3, 3);

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "There's a problem", Toast.LENGTH_SHORT).show();
        } else {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }
}