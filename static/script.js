document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const placeholder = document.getElementById('results-placeholder');
    const resultsCard = document.getElementById('results-card');

    // Inputs for real-time recalculation
    const sem1Input = document.getElementById('Sem1_Marks');
    const sem2Input = document.getElementById('Sem2_Marks');
    const sem3Input = document.getElementById('Sem3_Marks');
    const sem4Input = document.getElementById('Sem4_Marks');
    const y1BacklogsInput = document.getElementById('Year1_Backlogs');
    const y2BacklogsInput = document.getElementById('Year2_Backlogs');

    // Badges & trajectory elements
    const y1Badge = document.getElementById('y1-live-badge');
    const y2Badge = document.getElementById('y2-live-badge');
    const trajCgpa = document.getElementById('traj-cgpa');
    const trajTrend = document.getElementById('traj-trend');
    const trajBacklogs = document.getElementById('traj-backlogs');

    // Results elements
    const resProgramTitle = document.getElementById('res-program-title');
    const resRiskBadge = document.getElementById('res-risk-badge');
    const resPerfScore = document.getElementById('res-perf-score');
    const resHonours = document.getElementById('res-honours');
    const resCgpa = document.getElementById('res-cgpa');
    const resDropoutStatus = document.getElementById('res-dropout-status');
    const resDropoutProb = document.getElementById('res-dropout-prob');

    const resY1Avg = document.getElementById('res-y1-avg');
    const resY2Avg = document.getElementById('res-y2-avg');
    const resCumMarks = document.getElementById('res-cum-marks');
    const resTrend = document.getElementById('res-trend');
    const resTotalBacklogs = document.getElementById('res-total-backlogs');

    const facultyList = document.getElementById('faculty-solutions-list');
    const studentList = document.getElementById('student-solutions-list');
    const riskList = document.getElementById('risk-factors-list');

    // Preset buttons
    const btnPresetPass = document.getElementById('preset-pass');
    const btnPresetRisk = document.getElementById('preset-risk');

    // Live calculation function
    function updateLiveCalculations() {
        const s1 = parseFloat(sem1Input.value) || 0;
        const s2 = parseFloat(sem2Input.value) || 0;
        const s3 = parseFloat(sem3Input.value) || 0;
        const s4 = parseFloat(sem4Input.value) || 0;

        const y1Avg = (s1 + s2) / 2.0;
        const y2Avg = (s3 + s4) / 2.0;
        const cumAvg = (y1Avg + y2Avg) / 2.0;
        const cgpa = cumAvg / 10.0;
        const trend = y2Avg - y1Avg;

        const y1Bl = parseInt(y1BacklogsInput.value) || 0;
        const y2Bl = parseInt(y2BacklogsInput.value) || 0;
        const totalBl = y1Bl + y2Bl;

        // Update Year 1 & 2 live pills
        if (y1Badge) {
            y1Badge.innerHTML = `Year 1 Avg: <strong>${y1Avg.toFixed(1)}%</strong> (${(y1Avg / 10.0).toFixed(2)} SGPA)`;
        }
        if (y2Badge) {
            y2Badge.innerHTML = `Year 2 Avg: <strong>${y2Avg.toFixed(1)}%</strong> (${(y2Avg / 10.0).toFixed(2)} SGPA)`;
        }

        // Update Trajectory Banner
        if (trajCgpa) trajCgpa.textContent = `${cgpa.toFixed(2)} / 10.0`;

        if (trajTrend) {
            const sign = trend > 0 ? '+' : '';
            trajTrend.textContent = `${sign}${trend.toFixed(1)}%`;
            trajTrend.className = 'traj-val';
            if (trend > 2.0) {
                trajTrend.classList.add('trend-positive');
                trajTrend.textContent += ' (Improving)';
            } else if (trend < -5.0) {
                trajTrend.classList.add('trend-negative');
                trajTrend.textContent += ' (Declining)';
            } else {
                trajTrend.classList.add('trend-neutral');
                trajTrend.textContent += ' (Steady)';
            }
        }

        if (trajBacklogs) {
            trajBacklogs.textContent = `${totalBl} Subject${totalBl === 1 ? '' : 's'}`;
            trajBacklogs.className = 'traj-val';
            if (totalBl > 2) {
                trajBacklogs.classList.add('trend-negative');
            } else if (totalBl > 0) {
                trajBacklogs.classList.add('trend-neutral');
            } else {
                trajBacklogs.classList.add('trend-positive');
            }
        }
    }

    // Attach input listeners
    [sem1Input, sem2Input, sem3Input, sem4Input, y1BacklogsInput, y2BacklogsInput].forEach(inp => {
        if (inp) inp.addEventListener('input', updateLiveCalculations);
    });

    // Run once on initial load
    updateLiveCalculations();

    // Presets
    if (btnPresetPass) {
        btnPresetPass.addEventListener('click', function () {
            document.getElementById('Course_Program').value = 'Computer Science & Engineering';
            sem1Input.value = '82.0';
            sem2Input.value = '85.0';
            sem3Input.value = '84.0';
            sem4Input.value = '87.5';
            y1BacklogsInput.value = '0';
            y2BacklogsInput.value = '0';
            document.getElementById('Attendance_Percentage').value = '92';
            document.getElementById('Study_Hours_Per_Week').value = '25';
            document.getElementById('Tutoring_Sessions').value = '2';
            document.getElementById('Previous_Academic_Score').value = '88.0';
            document.getElementById('Motivation_Level').value = 'High';
            document.getElementById('Access_to_Resources').value = 'High';
            updateLiveCalculations();
        });
    }

    if (btnPresetRisk) {
        btnPresetRisk.addEventListener('click', function () {
            document.getElementById('Course_Program').value = 'Mechanical Engineering';
            sem1Input.value = '58.0';
            sem2Input.value = '52.0';
            sem3Input.value = '42.0';
            sem4Input.value = '39.0';
            y1BacklogsInput.value = '1';
            y2BacklogsInput.value = '3';
            document.getElementById('Attendance_Percentage').value = '61';
            document.getElementById('Study_Hours_Per_Week').value = '8';
            document.getElementById('Tutoring_Sessions').value = '0';
            document.getElementById('Previous_Academic_Score').value = '52.0';
            document.getElementById('Motivation_Level').value = 'Low';
            document.getElementById('Access_to_Resources').value = 'Low';
            updateLiveCalculations();
        });
    }

    // Tab Switching
    window.switchTab = function (tabName) {
        const tabFaculty = document.getElementById('tab-faculty');
        const tabStudent = document.getElementById('tab-student');
        const btnFaculty = document.getElementById('tab-btn-faculty');
        const btnStudent = document.getElementById('tab-btn-student');

        if (tabName === 'faculty') {
            tabFaculty.classList.remove('hidden');
            tabStudent.classList.add('hidden');
            btnFaculty.classList.add('active');
            btnStudent.classList.remove('active');
        } else {
            tabFaculty.classList.add('hidden');
            tabStudent.classList.remove('hidden');
            btnFaculty.classList.remove('active');
            btnStudent.classList.add('active');
        }
    };

    // Form Submission
    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const submitBtn = document.getElementById('submit-btn');
        submitBtn.disabled = true;
        submitBtn.querySelector('.btn-text').textContent = 'Analyzing 2-Year Progression...';

        const formData = new FormData(form);
        const data = {};

        formData.forEach((value, key) => {
            if (!isNaN(value) && value.trim() !== '') {
                data[key] = Number(value);
            } else {
                data[key] = value;
            }
        });

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Prediction API returned an error');
            }
            return response.json();
        })
        .then(result => {
            // Update Title & Badges
            resProgramTitle.textContent = data.Course_Program || 'Engineering Department';

            resRiskBadge.textContent = result.risk_level;
            resRiskBadge.className = 'badge-risk';
            if (result.risk_level === 'High Risk') {
                resRiskBadge.classList.add('badge-high');
            } else if (result.risk_level === 'Moderate Risk') {
                resRiskBadge.classList.add('badge-moderate');
            } else {
                resRiskBadge.classList.add('badge-low');
            }

            // Top Metric Cards
            resPerfScore.textContent = result.performance_prediction;
            resHonours.textContent = result.academic_division;
            resCgpa.textContent = result.predicted_cgpa;
            resDropoutStatus.textContent = result.dropout_prediction;

            if (result.dropout_prediction === 'Dropout') {
                resDropoutStatus.className = 'metric-value text-rose';
            } else if (result.dropout_prediction === 'Enrolled') {
                resDropoutStatus.className = 'metric-value text-indigo';
            } else {
                resDropoutStatus.className = 'metric-value text-emerald';
            }

            if (result.dropout_probability !== null && result.dropout_probability !== undefined) {
                resDropoutProb.textContent = `Estimated Dropout Risk: ${result.dropout_probability}%`;
            } else {
                resDropoutProb.textContent = '';
            }

            // Analytics Strip
            const an = result.analytics_summary;
            if (an) {
                resY1Avg.textContent = an.year1_average;
                resY2Avg.textContent = an.year2_average;
                resCumMarks.textContent = an.overall_2year_marks;
                resTrend.textContent = an.progression_trend;
                resTotalBacklogs.textContent = `${an.total_backlogs} (Y1: ${an.year1_backlogs}, Y2: ${an.year2_backlogs})`;
            }

            // Populate Faculty Solutions
            facultyList.innerHTML = '';
            if (result.faculty_solutions && result.faculty_solutions.length > 0) {
                result.faculty_solutions.forEach(sol => {
                    const li = document.createElement('li');
                    li.textContent = sol;
                    facultyList.appendChild(li);
                });
            }

            // Populate Student Solutions
            studentList.innerHTML = '';
            if (result.student_solutions && result.student_solutions.length > 0) {
                result.student_solutions.forEach(sol => {
                    const li = document.createElement('li');
                    li.textContent = sol;
                    studentList.appendChild(li);
                });
            }

            // Populate Risk Factors
            riskList.innerHTML = '';
            if (result.risk_factors && result.risk_factors.length > 0) {
                result.risk_factors.forEach(factor => {
                    const li = document.createElement('li');
                    li.textContent = factor;
                    riskList.appendChild(li);
                });
            }

            // Reveal results
            if (placeholder) placeholder.classList.add('hidden');
            resultsCard.classList.remove('hidden');
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        })
        .catch(err => {
            console.error('Error running prediction:', err);
            alert('An error occurred during engineering evaluation. Please check server status.');
        })
        .finally(() => {
            submitBtn.disabled = false;
            submitBtn.querySelector('.btn-text').textContent = 'Run Engineering Analysis & Predict';
        });
    });
});
