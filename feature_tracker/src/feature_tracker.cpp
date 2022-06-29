#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    /**
     * @brief     { Checks if point is in border }
     * @param     pt   point to be checked if in border of image or not (in border means inside the image)
     * @return    true if the point is in the border of the image
     */
    const int BORDER_SIZE = 1; // in pixel for (1, 1)

    // round to nearest integer
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    /**
     * @brief    { Reduce the vector v to the elements with status 1 }
     * @param    v        vector to be reduced
     * @param    status   vector of status of the vector v (1 for keeping, 0 for deleting)
     */
    int j = 0;
    for (int i = 0; i < int(v.size()); i++) // for each element in v
        if (status[i]) // if status is 1
            v[j++] = v[i]; // keep the element
    v.resize(j); // resize v to the number of elements with status 1
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    /**
     * @brief    { Reduce the vector v to the elements with status 1 }
     * @param    v         vector to be reduced
     * @param    status    vector of status of the vector v (1 for keeping, 0 for deleting)
     */
    int j = 0;
    for (int i = 0; i < int(v.size()); i++) // for each element in v
        if (status[i]) // if status is 1
            v[j++] = v[i]; // keep the element
    v.resize(j); // resize v to the number of elements with status 1
}


FeatureTracker::FeatureTracker() { }

void FeatureTracker::setMask()
{
    /**
     * @brief    { Set the mask for the feature tracker }
     */
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // initialize mask

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++) // push back long time tracked points and ids
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // sort the points by the track_cnt in descending order
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    // clear vectors
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    // mask based on the points with the largest track_cnt
    for (auto &it : cnt_pts_id)
    {
        // check only pixels that have not been masked yet
        if (mask.at<uchar>(it.second.first) == 255)
        {
            // push back points, ids and track_cnt that meet the criteria (remove dense points)
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    /**
     * @brief    { Add new features point to existing forward points }
     */
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    /**
     * @brief    { Reads the image from the topic }
     * @param    _img         image from the topic
     * @param    _cur_time    current time of the image
     */
    cv::Mat img; // image to be processed
    TicToc t_r; // time to read image
    cur_time = _cur_time;  // current time of the image

    if (EQUALIZE) // param EQUALIZE: if image is too dark or light, turn on equalize to find enough features
    {
        // create CLAHE object for equalize
        // clipLimit: number of pixels in each tile
        // tileGridSize: size of the tile grid for each dimension (x, y)
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img); // equalize the image
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc()); // print time to equalize
    }
    else
        img = _img;

    if (forw_img.empty()) // if the first image
    {
        prev_img = cur_img = forw_img = img; // set images to be the first image
    }
    else
    {
        forw_img = img; // update the forward image
    }

    forw_pts.clear(); // clear the points of the forward image

    if (cur_pts.size() > 0) // if there are points in the current image
    {
        TicToc t_o;
        vector<uchar> status; // status of the points in the current image
        vector<float> err; // error of the points in the current image
        // calculate optical flow between the current image and the previous image (output: nextPts, status, err)
        // prevImg:	first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
        // nextImg: second input image or pyramid of the same size and the same type as prevImg.
        // prevPts:	vector of 2D points for which the flow needs to be found;
        //          point coordinates must be single-precision floating-point numbers.
        // nextPts: output vector of 2D points (with single-precision floating-point coordinates)
        //          containing the calculated new positions of input features in the second image;
        //          when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
        // status: output status vector (of unsigned chars);
        //         each element of the vector is set to 1 if the flow for the corresponding features has been found,
        //         otherwise, it is set to 0.
        // err:    output vector of errors; each element of the vector is set to an error for the corresponding feature,
        //         type of the error measure can be set in flags parameter;
        //         if the flow wasn't found then the error is not defined (use the status parameter to find such cases).
        // winSize: size of the search window at each pyramid level.
        // maxLevel: 0-based maximal pyramid level number;
        //           if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on;
        //           if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel.
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0; // if not in the border, set the status to 0

        // reduce the size of the vector according to the status vector
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt) // update the track count of the points
        n++; // increase the track count of the points

    if (PUB_THIS_FRAME)
    {
        rejectWithF(); // reject points with low feature response
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask(); // set the mask based on frequent feature point
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, // input image
                                    n_pts, // output vector of detected points
                                    n_max_cnt, // maximum number of points to be returned
                                    0.01, // minimal accepted quality of image corners
                                    MIN_DIST, // minimum possible Euclidean distance between the returned points
                                    mask); // optional region of interest
        }
        else // if there are too many points in the current image
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints(); // add new features point to existing forw_pts
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    // update values and get undistorted normalized points & velocity
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints(); // get undistorted normalized points and velocity
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    /**
    * @brief    { Rejects points with low feature response }
    */
    if (forw_pts.size() >= 8) // if there are 8 points or more for RANSAC
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        // create undistorted points (pixel coordinates)
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p; // temporary undistorted point
            // liftProjective(pixel coordinates, world coordinates)
            // undistortion of the points (pixel coordinates to world coordinates)
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // undistorted point from world coordinates to pixel coordinates
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y()); // undistorted points (pixel coordinates)

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status; // status of the inliers
        // findFundamentalMat(...) for removing outliers
        // input: point1, point2
        // output: status vector (uchar)
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    /**
     * @brief    { update the id of feature point i }
     * @param    i    the index of feature point in cur_pts
     * @return        true if the id is updated, false otherwise
     */
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++; // new feature point
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    /**
    * @brief    { get undistorted normalized points and velocity }
    */
    cur_un_pts.clear(); // clear the last vectors
    cur_un_pts_map.clear(); // clear the last vectors
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y); //
        Eigen::Vector3d b;
        // undistortion of the points (pixel coordinates to world coordinates)
        m_camera->liftProjective(a, b);
        // push back normalized points
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        // push back ids and normalized points
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // calculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time; // time interval
        pts_velocity.clear(); // clear the last vectors
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                // find a previous point that matches the id of the current point
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map; // update current points to previous points
}
