$(function() {
  const $unlock = $('#unlockBtn');
  const $lock = $('#lockBtn');
  const $statusText = $('#statusText');
  const $video = $('#video');
  const $overlay = $('#resultOverlay');
  const $bigResult = $('#bigResult');
  const $bigMsg = $('#bigResultMsg');
  const $proceed = $('#proceedBtn');

  let lastLogId = null;
  let pollTimer = null;

  function startPollingLastLog(){
    if(pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(()=>{
      fetch('/last_log').then(r=>r.json()).then(data=>{
        if(data && data.id && data.id !== lastLogId){
          lastLogId = data.id;
          // show overlay with result
          showResultOverlay(data.result, data.image_path);
          // stop polling after result shown
          clearInterval(pollTimer);
        }
      }).catch(()=>{});
    }, 600);
  }

  function showResultOverlay(result, imgPath){
    $bigResult.text(result.toUpperCase());
    if(result.toUpperCase() === 'REAL'){
      $bigResult.css('color','limegreen');
      $bigMsg.text('Verified — you may proceed.');
      $proceed.attr('href','/'); // change to protected page if any
    } else {
      $bigResult.css('color','crimson');
      $bigMsg.text('Spoof Detected — Access Denied.');
      $proceed.attr('href','#');
    }
    $overlay.show();
    // video will automatically be off because server sets STREAMING False
    $unlock.removeClass('d-none');
    $lock.addClass('d-none');
    $statusText.text('Locked — click Unlock to verify');
  }

  $unlock.on('click', function() {
    fetch('/start_stream', { method: 'POST' })
      .then(r => r.json())
      .then(() => {
        // switch buttons
        $unlock.addClass('d-none');
        $lock.removeClass('d-none');
        $statusText.text('Camera ON — Align your face inside the box');
        // reload stream (cache-bust)
        $video.attr('src', '/video_feed?' + new Date().getTime());
        // start polling for completed result
        startPollingLastLog();
      });
  });

  $lock.on('click', function() {
    fetch('/stop_stream', { method: 'POST' })
      .then(r => r.json())
      .then(() => {
        $lock.addClass('d-none');
        $unlock.removeClass('d-none');
        $statusText.text('Locked — click Unlock to verify');
        $video.attr('src', '/video_feed?' + new Date().getTime());
        if(pollTimer) clearInterval(pollTimer);
      });
  });

  $('#closeOverlay').on('click', function(){
    $overlay.hide();
    $video.attr('src', '/video_feed?' + new Date().getTime());
  });

  // Proceed button behaviour - you can change destination to protected route
  $proceed.on('click', function(e){
    if($bigResult.text().toUpperCase() === 'REAL'){
      // redirect to a protected page or show content
      window.location.href = '/'; // replace with actual protected route
    } else {
      e.preventDefault();
      alert('Access denied.');
    }
  });

});
