// UDOS_seminarski_meanshift+SIFT.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace cv;
using namespace std;



int main(int argc, char** argv)
{
	const string about =
		"This sample demonstrates the meanshift algorithm.\n"
		"The example file can be downloaded from:\n"
		" https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4";
	const string keys =
		"{ h help | | print this help message }"
		"{ @image |<none>| path to image file }";
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	string filename = "D:/Desktop/video.mp4"; //ucitavanje videa - odabrati pravi path
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	VideoCapture capture(filename);
	if (!capture.isOpened()) {
		cerr << "Unable to open file!" << endl;
		return 0;
	}
	Mat frame, roi, hsv_roi, mask;

	capture >> frame;





	Rect track_window;// prozor od interesa
	float range_[] = { 0, 180 };
	const float* range[] = { range_ };
	Mat roi_hist;
	int histSize[] = { 180 };
	int channels[] = { 0 };
	TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);

	int sizex, sizey;
	int frames_cnt = 0;
	Mat imgMatches;
	bool var = 0;
	int frames_detected_ms = 0;
	int cnt = 0;
	while (true) {
		Mat hsv, dst;
		capture >> frame;

		if (frame.empty())
			break;


		Mat mask1;
		Mat image;
		Mat gray;
		image = frame.clone();

		mask1 = cv::imread("D:/Desktop/sivoauto.jpg", cv::IMREAD_COLOR); //template za detekciju - odabrati pravi path
		if (mask1.empty()) {
			std::cout << "Prazna slike";
			return 1;
		}
		cv::cvtColor(mask1, mask1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

		Ptr<SIFT> sift = SIFT::create();
		Ptr<SIFT> sift1 = SIFT::create();
		vector<KeyPoint>keypoints, keypoints1;
		Mat deskriptori, deskriptori1;
		sift->detectAndCompute(gray, noArray(), keypoints, deskriptori);
		sift1->detectAndCompute(mask1, noArray(), keypoints1, deskriptori1);
		Mat output, output1;


		BFMatcher matcher(cv::NORM_L2);
		vector<DMatch> mech;
		vector<DMatch> matches;

		matcher.match(deskriptori, deskriptori1, mech);
		double maxdistance = mech[0].distance;
		double distance_curr;
		for (int i = 1; i < mech.size(); i++) {
			distance_curr = mech[i].distance;
			if (distance_curr > maxdistance) {
				maxdistance = distance_curr;
			}
		}
		double k = 0.2;

		for (int i = 0; i < mech.size(); i++) {
			if (mech[i].distance <= k * maxdistance) {
				matches.push_back(mech[i]);
			}
		}


		if ((matches.size() >= 0 && matches.size() <= 10 && var==0) || cnt>=10) { //objekat nije detektovan
			if (cnt >= 10) {
				cnt = 0;
				var = 0;
				std::cout << std::endl << "cnt je vec od 10 " << std::endl;
				break;
			}
			std::cout << endl << "DOSLO u matches 0" << endl;
			putText(frame, "Objekat nije detektovan", Point(50, 20), 5, 1, Scalar(0, 0, 0));

		}
		else if (matches.size() > 10 || var == 1) { //objekat je detektovan
			if (matches.size() >= 0 && matches.size() <= 3) {
				cnt++;
			}
			var = 1;
			std::cout << endl << "DOSLO u else" << endl;
			frames_cnt++;
			if (frames_cnt == 5) {

				//crtanje prvog kvadrat za meanshift

				vector<Point> coordinates;
				vector<int> indexes;
				for (int i = 0; i < matches.size(); i++) {
					int idx = matches[i].queryIdx;
					indexes.push_back(idx);
					coordinates.push_back(keypoints[idx].pt);
				}

				Rect initial = boundingRect(coordinates); //kvadrat koji se koristi za provjeru da li je MS ili CS ispravno lokalizirao objekat



				sizex = 100;
				sizey = 50;
				track_window = Rect(initial.x, initial.y, sizex, sizey); 
				roi = frame(track_window);
				std::cout << endl << "Prvi frejm sa motorom" << endl;
				cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
				inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);

				calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
				normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);
			}
			else if (frames_cnt > 5) { //detekcija je malo pomjerena iz razloga da se kvadrati ne bi crtali preko granica slike

				std::cout << endl << "ostali frejmovi sa motorom" << endl;
				cvtColor(frame, hsv, COLOR_BGR2HSV);
				calcBackProject(&hsv, 1, channels, roi_hist, dst, range);
				
				//meanShift(dst, track_window, term_crit); // u slucaju koristenja MS ovu liniju odkomentarisati

				//u slucaju koristenja CS donjih 5 linija odkomentarisati
				//RotatedRect rot_rect = CamShift(dst, track_window, term_crit);
				//Point2f points[4];
				//rot_rect.points(points);
				//for (int i = 0; i < 4; i++)
				//	line(frame, points[i], points[(i + 1) % 4], 255, 2);
				//dovde

				if (track_window.br().x < frame.cols && track_window.tl().x + sizex < frame.cols) { //provjera da li kvadrat fituje u sliku
						
					rectangle(frame, track_window, 255, 2);

					vector<Point> coordinates1;
					vector<int> indexes1;
					for (int i = 0; i < matches.size(); i++) {
						int idx = matches[i].queryIdx;
						indexes1.push_back(idx);
						coordinates1.push_back(keypoints[idx].pt);
					}

					Rect initial1 = boundingRect(coordinates1);
					rectangle(frame, initial1, Scalar(0, 0, 255));
					if (track_window.contains(initial1.br()) || track_window.contains(initial1.tl()) || initial1.contains(track_window.br()) || initial1.contains(track_window.tl())) {
						frames_detected_ms++;
				    }

				}
			}
		}

		//SIFT KRAJ

		imshow("img2", frame);

		int keyboard = waitKey(1);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}
	std::cout << endl << " Broj frejmova sa sivim autom (tacan broj frejmova dobiven SIFTom): "<<frames_cnt << endl;
	std::cout << endl << " Broj frejmova sa sivim autom (dobiven mean shiftom): " << frames_detected_ms << endl;
}

