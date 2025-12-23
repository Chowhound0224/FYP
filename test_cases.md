# Test Cases - AI Resume Screening System

## Document Information
- **Project:** AI Resume Screening System
- **Version:** 1.0
- **Date:** December 2024
- **Test Environment:** Streamlit Web Application

---

## Test Case Categories

1. [Job Seeker Test Cases](#job-seeker-test-cases)
2. [HR/Recruiter Test Cases](#hrrecruiter-test-cases)
3. [System Integration Test Cases](#system-integration-test-cases)
4. [Error Handling Test Cases](#error-handling-test-cases)
5. [UI/UX Test Cases](#uiux-test-cases)

---

## Job Seeker Test Cases

### TC-JS-001: Upload Single Resume (PDF)
**Objective:** Verify that a job seeker can upload a single PDF resume successfully

**Preconditions:**
- Application is running
- User is on the Resume Upload page
- Valid PDF resume file is available

**Test Steps:**
1. Navigate to "📤 Resume Upload" page
2. Click on the file uploader or drag & drop area
3. Select a valid PDF resume file
4. Click "🚀 Analyze Resumes" button
5. Wait for processing to complete

**Expected Results:**
- File uploads successfully with green success message
- Progress bar shows processing status
- AI prediction results display with:
  - Predicted category badge
  - Confidence score
  - Top 5 category scores with progress bars
  - Text extraction preview
- Summary statistics show (Average Certainty, Unique Categories, Strong Classifications)
- CSV download button is available

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-JS-002: Upload Multiple Resumes
**Objective:** Verify that multiple resumes can be uploaded and analyzed together

**Preconditions:**
- Application is running
- User is on the Resume Upload page
- Multiple resume files (3-5 files) are available in different formats

**Test Steps:**
1. Navigate to "📤 Resume Upload" page
2. Select multiple files (PDF, DOCX, TXT) using Ctrl+Click (Windows) or Cmd+Click (Mac)
3. Click "🚀 Analyze Resumes" button
4. Observe processing for all files

**Expected Results:**
- All files upload successfully
- Success message shows correct count: "✅ X file(s) uploaded successfully!"
- Each file is processed individually with progress indicator
- Individual prediction cards appear for each resume
- All files appear in the results section
- Summary statistics reflect all uploaded resumes

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-JS-003: Upload DOCX Resume
**Objective:** Verify DOCX file format is supported and processed correctly

**Preconditions:**
- Valid DOCX resume file is available

**Test Steps:**
1. Navigate to Resume Upload page
2. Upload a DOCX resume file
3. Click "🚀 Analyze Resumes" button

**Expected Results:**
- DOCX file is accepted
- Text extraction successful
- Prediction results display correctly
- Text preview shows extracted content

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-JS-004: Upload TXT Resume
**Objective:** Verify TXT file format is supported and processed correctly

**Preconditions:**
- Valid TXT resume file is available

**Test Steps:**
1. Navigate to Resume Upload page
2. Upload a TXT resume file
3. Click "🚀 Analyze Resumes" button

**Expected Results:**
- TXT file is accepted
- Text extraction successful
- Prediction results display correctly
- Full text content is analyzed

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-JS-005: View Detailed Analysis
**Objective:** Verify detailed analysis can be expanded and viewed

**Preconditions:**
- At least one resume has been analyzed

**Test Steps:**
1. Upload and analyze a resume
2. Locate the prediction card in results
3. Click on "📊 Detailed Analysis for [filename]" expander

**Expected Results:**
- Expander opens smoothly
- Left column shows "Top 5 Category Scores" with progress bars
- Predicted category is bolded
- Right column shows "Text Extraction Preview"
- Preview text is readable (max 700 characters)

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-JS-006: Export Results to CSV
**Objective:** Verify prediction results can be exported to CSV

**Preconditions:**
- At least one resume has been analyzed

**Test Steps:**
1. Upload and analyze resume(s)
2. Scroll to "📥 Export Results" section
3. Click "📥 Download Results (CSV)" button
4. Open downloaded CSV file

**Expected Results:**
- CSV file downloads successfully
- File name is "resume_predictions.csv"
- CSV contains columns: Filename, Predicted Category, Confidence
- All analyzed resumes appear in CSV
- Confidence values are formatted as percentages (e.g., "85.50%")

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-JS-007: Navigate Back to Home
**Objective:** Verify navigation from Resume Upload page to home

**Preconditions:**
- User is on Resume Upload page

**Test Steps:**
1. Scroll to bottom of page
2. Click "🏠 Back to Home" button

**Expected Results:**
- Application navigates to landing page
- Landing page displays two action cards
- No errors occur during navigation

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

## HR/Recruiter Test Cases

### TC-HR-001: Define Job Requirements - Complete Flow
**Objective:** Verify HR can define complete job requirements

**Preconditions:**
- Application is running
- User is on Job Matching page

**Test Steps:**
1. Navigate to "🎯 Job Matching" page
2. In "Target Job Category" section:
   - Select "ENGINEERING" from dropdown
3. In "Job Description" field:
   - Enter: "We are looking for a Software Engineer with 5+ years experience in Python, Java, and cloud technologies."
4. In "Required Keywords" field:
   - Enter: "python, java, aws, kubernetes, docker"
5. Click "💾 Save Job Requirements" button

**Expected Results:**
- Success message appears: "✅ Job requirements saved! Target category: ENGINEERING with 5 keyword(s)"
- Info box displays saved requirements
- System is ready to accept candidate uploads

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-002: Define Job Requirements - Custom Category
**Objective:** Verify custom category can be entered

**Preconditions:**
- User is on Job Matching page

**Test Steps:**
1. Navigate to Job Matching page
2. Leave dropdown at default
3. Enter "DATA-SCIENTIST" in custom category field
4. Enter job description
5. Click "💾 Save Job Requirements"

**Expected Results:**
- Custom category is accepted
- Success message shows: "Target category: DATA-SCIENTIST"
- System saves custom category correctly

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-003: Upload Candidate Resumes
**Objective:** Verify HR can upload multiple candidate resumes

**Preconditions:**
- Job requirements have been defined
- Multiple candidate resume files are available

**Test Steps:**
1. After defining job requirements
2. Scroll to "📤 Step 2: Upload Candidate Resumes"
3. Upload 3-5 candidate resumes
4. Verify files are uploaded
5. Click "🚀 Rank Candidates" button

**Expected Results:**
- Upload area accepts multiple files
- Success message shows: "✅ X resume(s) ready for analysis"
- Progress bar shows during ranking
- Each candidate is analyzed with status update

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-004: View Candidate Rankings
**Objective:** Verify candidate rankings are displayed correctly

**Preconditions:**
- Candidates have been ranked

**Test Steps:**
1. After ranking completes, scroll to "🏆 Candidate Rankings"
2. Review summary statistics (4 metrics)
3. Review summary table
4. Review detailed candidate cards

**Expected Results:**
- Summary shows:
  - Total Candidates
  - Perfect Matches
  - Avg Match Score
  - Strong Candidates
- Summary table displays with columns: Rank, Filename, Match Score, Predicted Category, Category Match
- High-scoring candidates highlighted in green (#0A9548)
- Medium scores highlighted in yellow
- Top 3 candidates show medal icons (🥇🥈🥉)

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-005: View Match Score Breakdown
**Objective:** Verify hybrid scoring breakdown is displayed

**Preconditions:**
- Candidates have been ranked

**Test Steps:**
1. Locate any candidate card
2. Click "📊 View Full Analysis" expander
3. Review "Score Breakdown (Hybrid Scoring)" section

**Expected Results:**
- Four score components displayed:
  - Category Match (30%): Shows points earned
  - Keywords (30%): Shows points earned
  - Content Similarity (30%): Shows points earned
  - Model Confidence (10%): Shows points earned
- Total Score shown: "Total Score: X/100"
- Values are calculated correctly

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-006: View Keyword Matches
**Objective:** Verify keyword matching results are displayed

**Preconditions:**
- Keywords were specified in job requirements
- Candidates have been ranked

**Test Steps:**
1. Open candidate's full analysis
2. Locate "🔑 Keyword Matches" section

**Expected Results:**
- Shows count: "X/Y keywords found"
- If matches found: Displays matched keywords as code blocks (e.g., `python` `aws`)
- If no matches: Shows "❌ 0/Y keywords found"
- Matches are case-insensitive

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-007: View Content Similarity
**Objective:** Verify SBERT similarity score is displayed

**Preconditions:**
- Candidates have been ranked

**Test Steps:**
1. Open candidate's full analysis
2. Locate "🔍 Content Similarity" section

**Expected Results:**
- Similarity percentage displayed (0-100%)
- Label shows: "Content Similarity: X% (resume vs. job description)"
- Value is based on SBERT embedding comparison

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-008: Export Rankings to CSV
**Objective:** Verify rankings can be exported to CSV

**Preconditions:**
- Candidates have been ranked

**Test Steps:**
1. Scroll to "📥 Export Rankings" section
2. Click "📥 Download Rankings (CSV)" button
3. Open downloaded CSV file

**Expected Results:**
- CSV downloads with filename: "candidate_rankings_[CATEGORY].csv"
- CSV contains columns: Rank, Filename, Match Score, Predicted Category, Confidence, Category Match, Keywords Matched
- All ranked candidates appear in CSV
- Data matches displayed rankings
- Keyword matches formatted as "X/Y (keyword1, keyword2)"

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-HR-009: Define Requirements Without Keywords
**Objective:** Verify system works when keywords are optional

**Preconditions:**
- User is on Job Matching page

**Test Steps:**
1. Select target category
2. Enter job description
3. Leave keywords field empty
4. Save job requirements
5. Upload and rank candidates

**Expected Results:**
- System accepts requirements without keywords
- Success message shows: "Target category: X" (without keyword count)
- Ranking completes successfully
- Keyword section shows "N/A" in rankings
- Hybrid scoring uses 0% for keyword component

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

## System Integration Test Cases

### TC-INT-001: End-to-End Resume Screening Flow
**Objective:** Test complete job seeker workflow

**Test Steps:**
1. Start application
2. Click "📂 Start Screening" on landing page
3. Upload 5 different resumes (mixed formats)
4. Analyze all resumes
5. Review all results
6. Export to CSV
7. Return to home

**Expected Results:**
- All steps complete without errors
- All 5 resumes analyzed successfully
- Results are accurate and complete
- CSV export works
- Navigation is smooth

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-INT-002: End-to-End Job Matching Flow
**Objective:** Test complete HR workflow

**Test Steps:**
1. Start application
2. Click "⭐ Find Matches" on landing page
3. Define job requirements with all fields
4. Upload 10 candidate resumes
5. Rank all candidates
6. Review top 3 candidates in detail
7. Export rankings to CSV
8. Return to home

**Expected Results:**
- All steps complete without errors
- All 10 candidates ranked successfully
- Rankings are sorted by match score (highest first)
- Top candidates show appropriate highlights
- CSV export includes all data
- Navigation works properly

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-INT-003: Model Prediction Accuracy
**Objective:** Verify AI model predicts categories correctly

**Preconditions:**
- Sample resumes with known categories available

**Test Steps:**
1. Upload resume for Software Engineer position
2. Note predicted category
3. Upload resume for Accountant position
4. Note predicted category
5. Upload resume for HR Manager position
6. Note predicted category

**Expected Results:**
- Software Engineer resume → ENGINEERING or INFORMATION-TECHNOLOGY
- Accountant resume → ACCOUNTANT or FINANCE
- HR Manager resume → HR
- Confidence scores > 50% for clear matches
- Top 5 categories show relevant alternatives

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-INT-004: Performance with Multiple Files
**Objective:** Test system performance with maximum files

**Test Steps:**
1. Navigate to Resume Upload page
2. Upload 20 resume files simultaneously
3. Click Analyze
4. Monitor processing time and system responsiveness

**Expected Results:**
- System accepts all 20 files
- Processing completes in reasonable time (< 5 minutes)
- Progress bar updates smoothly
- No crashes or freezes
- All results display correctly

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

## Error Handling Test Cases

### TC-ERR-001: Upload Corrupted PDF
**Objective:** Verify system handles corrupted PDF gracefully

**Preconditions:**
- Corrupted PDF file available

**Test Steps:**
1. Navigate to Resume Upload page
2. Upload corrupted PDF file
3. Click Analyze

**Expected Results:**
- Error message appears: "❌ [filename] - Corrupted or invalid PDF file. Error: [details]"
- File is skipped from analysis
- Summary shows: "⚠️ 1 file(s) skipped due to corruption..."
- System continues processing other files (if any)
- No crash occurs

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-002: Upload Empty File
**Objective:** Verify system detects and handles empty files

**Preconditions:**
- Empty PDF, DOCX, or TXT file available

**Test Steps:**
1. Navigate to Resume Upload page
2. Upload empty file
3. Click Analyze

**Expected Results:**
- Warning message appears: "⚠️ [filename] appears to be empty or contains no readable text."
- File is skipped from analysis
- Summary shows file was skipped
- Other files processed normally

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-003: Upload Unsupported File Format
**Objective:** Verify unsupported formats are rejected

**Preconditions:**
- File with unsupported extension (.jpg, .png, .xlsx) available

**Test Steps:**
1. Navigate to Resume Upload page
2. Try to upload unsupported file format

**Expected Results:**
- File uploader does not accept the file
- Only PDF, DOCX, TXT files can be selected
- No error occurs

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-004: Submit Without File Upload
**Objective:** Verify validation when no files uploaded

**Test Steps:**
1. Navigate to Resume Upload page
2. Click "🚀 Analyze Resumes" without uploading files

**Expected Results:**
- Info message shows: "👆 Upload one or more resume files to get started"
- Analyze button is not visible
- No errors occur

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-005: Submit Job Requirements Without Category
**Objective:** Verify validation for required fields

**Test Steps:**
1. Navigate to Job Matching page
2. Leave category fields empty
3. Enter job description
4. Click "💾 Save Job Requirements"

**Expected Results:**
- Error message: "⚠️ Please select a category from the dropdown or enter a custom category."
- Requirements not saved
- User can correct and resubmit

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-006: Submit Job Requirements Without Description
**Objective:** Verify job description is required

**Test Steps:**
1. Navigate to Job Matching page
2. Select category
3. Leave job description empty
4. Click "💾 Save Job Requirements"

**Expected Results:**
- Error message: "⚠️ Please enter a job description."
- Requirements not saved
- Form remains editable

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-007: Rank Without Job Requirements
**Objective:** Verify validation for ranking prerequisites

**Test Steps:**
1. Navigate to Job Matching page
2. Upload candidate resumes without defining job requirements

**Expected Results:**
- "🚀 Rank Candidates" button is not visible
- Info message shown: "👆 Upload resumes and save job requirements to view rankings."
- System guides user to complete prerequisites

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-008: Upload Corrupted DOCX
**Objective:** Verify corrupted DOCX handling

**Preconditions:**
- Corrupted DOCX file available

**Test Steps:**
1. Upload corrupted DOCX file
2. Attempt to analyze

**Expected Results:**
- Error message: "❌ [filename] - Corrupted or invalid DOCX file. Error: [details]"
- File skipped from analysis
- Summary shows skipped count
- No application crash

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-ERR-009: Upload Text File with Invalid Encoding
**Objective:** Verify handling of encoding errors

**Preconditions:**
- TXT file with unsupported encoding available

**Test Steps:**
1. Upload TXT file with invalid encoding
2. Attempt to analyze

**Expected Results:**
- Error message: "❌ [filename] - Unable to decode text file. File may be corrupted or in an unsupported encoding."
- File skipped gracefully
- Other files continue processing

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

## UI/UX Test Cases

### TC-UI-001: Dark Mode Display
**Objective:** Verify UI displays correctly in dark mode

**Preconditions:**
- System theme set to dark mode

**Test Steps:**
1. Set operating system to dark mode
2. Launch application
3. Navigate through all pages

**Expected Results:**
- Background color: Black (#000000)
- Text color: White (#FFFFFF)
- Cards: Dark gray (#1a1a1a)
- Accent color: Light blue (#A2D2FF)
- All text is readable
- Buttons have good contrast

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-002: Light Mode Display
**Objective:** Verify UI displays correctly in light mode

**Preconditions:**
- System theme set to light mode

**Test Steps:**
1. Set operating system to light mode
2. Launch application
3. Navigate through all pages

**Expected Results:**
- Background color: Light gray (#F5F5F5)
- Text color: Black (#000000)
- Cards: White (#FFFFFF)
- Accent color: Dark blue (#4A5FC1)
- All text is readable
- Good contrast throughout

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-003: Button Visibility and Contrast
**Objective:** Verify all buttons are visible and clickable

**Test Steps:**
1. Navigate to each page
2. Check all button elements
3. Verify button colors and text contrast

**Expected Results:**
- Start Screening button: Dark blue (#4A5FC1) with white text
- Find Matches button: Dark blue (#4A5FC1) with white text
- Analyze Resumes button: Dark blue, readable
- Rank Candidates button: Dark blue, readable
- All buttons have hover effects
- Text is clearly visible on all buttons

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-004: Page Title Consistency
**Objective:** Verify page titles have consistent styling

**Test Steps:**
1. Check landing page title
2. Check Resume Upload page title
3. Check Job Matching page title

**Expected Results:**
- All titles use same color: #667eea (purple-blue) or consistent color scheme
- Font sizes are appropriate
- Icons display correctly with titles
- Titles are center-aligned (Resume Upload) or left-aligned (Job Matching)

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-005: Responsive Layout
**Objective:** Verify layout adapts to different window sizes

**Test Steps:**
1. Start with full-screen window
2. Resize window to tablet size (768px)
3. Resize to mobile size (375px)
4. Check all pages

**Expected Results:**
- Columns stack appropriately on smaller screens
- Cards remain readable
- Buttons are accessible
- No horizontal scrolling on content
- All features remain functional

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-006: File Uploader Visual Feedback
**Objective:** Verify file uploader provides good visual feedback

**Test Steps:**
1. Navigate to upload page
2. Hover over file uploader
3. Drag file over uploader
4. Drop file

**Expected Results:**
- Uploader has dashed border with accent color
- Hover effect shows border color change and scale
- Drag-over shows visual feedback
- File names appear after upload
- Clear visual hierarchy

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-007: Progress Bar Visibility
**Objective:** Verify progress indicators are clear

**Test Steps:**
1. Upload and analyze files
2. Observe progress bar and status text

**Expected Results:**
- Progress bar is visible and smooth
- Status text updates for each file
- Shows "Processing X/Y: filename"
- Progress bar fills from 0% to 100%
- Bar color uses theme accent color

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-008: Color-Coded Results
**Objective:** Verify results use appropriate color coding

**Test Steps:**
1. Analyze resumes with varying confidence scores
2. Check result card colors
3. Check table row highlighting

**Expected Results:**
- High confidence (≥70%): Green border and background
- Medium confidence (50-70%): Yellow/orange border
- Low confidence (<50%): Red border
- Table rows: Green for high scores (#0A9548), yellow for medium
- Color coding is consistent and meaningful

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-009: Navigation Flow
**Objective:** Verify intuitive navigation between pages

**Test Steps:**
1. Start from landing page
2. Navigate to Resume Upload
3. Use "Back to Home" button
4. Navigate to Job Matching
5. Use "Back to Home" button

**Expected Results:**
- All navigation works smoothly
- No broken links
- Page state is maintained appropriately
- Back buttons work correctly
- Landing page always accessible

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

### TC-UI-010: Medal Icons for Top Rankings
**Objective:** Verify medal icons display for top 3 candidates

**Test Steps:**
1. Rank at least 3 candidates
2. Check detailed candidate cards

**Expected Results:**
- Rank 1 shows: 🥇 #1
- Rank 2 shows: 🥈 #2
- Rank 3 shows: 🥉 #3
- Ranks 4+ show: #4, #5, etc.
- Icons are clearly visible

**Status:** [ ] Pass [ ] Fail

**Notes:**

---

## Test Execution Summary

**Total Test Cases:** 50
- Job Seeker: 7
- HR/Recruiter: 9
- Integration: 4
- Error Handling: 9
- UI/UX: 10
- System: 11

**Passed:** _____
**Failed:** _____
**Blocked:** _____
**Not Executed:** _____

**Overall Status:** [ ] Pass [ ] Fail

---

## Test Sign-Off

**Tester Name:** ________________________

**Date:** ________________________

**Signature:** ________________________

**Comments:**

---

## Appendix A: Test Data Requirements

### Sample Resumes Needed:
1. Software Engineer resume (PDF)
2. Accountant resume (DOCX)
3. HR Manager resume (TXT)
4. Designer resume (PDF)
5. Sales resume (DOCX)
6. Corrupted PDF file
7. Empty file (any format)
8. Text file with invalid encoding

### Job Requirements Samples:
1. Engineering position with 5 keywords
2. Finance position with 3 keywords
3. HR position without keywords
4. Custom category position

---

## Appendix B: Known Issues

(Document any known issues discovered during testing)

---

## Appendix C: Browser Compatibility

Test in the following browsers:
- [ ] Google Chrome (Latest)
- [ ] Mozilla Firefox (Latest)
- [ ] Microsoft Edge (Latest)
- [ ] Safari (Latest)

---

**End of Test Cases Document**
