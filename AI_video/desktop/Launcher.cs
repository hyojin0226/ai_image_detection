// A single executable containing an isolated Python runtime and the frozen model.
// The appended archive is extracted once per content hash, without administrator rights.
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

internal sealed class ArchiveSlice : Stream {
    private readonly Stream source;
    private readonly long start, length;
    public ArchiveSlice(Stream stream, long offset, long count) { source=stream; start=offset; length=count; Position=0; }
    public override bool CanRead { get { return true; } }
    public override bool CanSeek { get { return true; } }
    public override bool CanWrite { get { return false; } }
    public override long Length { get { return length; } }
    public override long Position { get { return source.Position-start; } set { if(value<0 || value>length) throw new IOException(); source.Position=start+value; } }
    public override int Read(byte[] buffer, int offset, int count) { return source.Read(buffer,offset,(int)Math.Min(count,length-Position)); }
    public override long Seek(long offset, SeekOrigin origin) { Position=(origin==SeekOrigin.Begin?0:origin==SeekOrigin.Current?Position:length)+offset; return Position; }
    public override void Flush() { }
    public override void SetLength(long value) { throw new NotSupportedException(); }
    public override void Write(byte[] buffer,int offset,int count) { throw new NotSupportedException(); }
    protected override void Dispose(bool disposing) { if(disposing) source.Dispose(); base.Dispose(disposing); }
}

internal sealed class ArchiveParts : Stream {
    private readonly FileStream[] files;
    private readonly long[] offsets;
    private long position, length;
    public ArchiveParts(string folder,int count) {
        files=new FileStream[count]; offsets=new long[count];
        for(int i=0;i<count;i++) {
            offsets[i]=length;
            string name=Path.Combine(folder,"runtime."+(i+1).ToString("D3"));
            if(!File.Exists(name)) throw new IOException("실행 환경 파일이 빠져 있습니다. GitHub에서 AI_video 폴더 전체를 받아 주세요.\n"+name);
            files[i]=File.OpenRead(name); length+=files[i].Length;
        }
    }
    public override bool CanRead { get { return true; } }
    public override bool CanSeek { get { return true; } }
    public override bool CanWrite { get { return false; } }
    public override long Length { get { return length; } }
    public override long Position { get { return position; } set { if(value<0 || value>length) throw new IOException(); position=value; } }
    public override int Read(byte[] buffer,int offset,int count) {
        int total=0;
        while(count>0 && position<length) {
            int index=files.Length-1;
            while(index>0 && offsets[index]>position) index--;
            FileStream file=files[index]; file.Position=position-offsets[index];
            int read=file.Read(buffer,offset,(int)Math.Min(count,file.Length-file.Position));
            if(read==0) break;
            total+=read; position+=read; offset+=read; count-=read;
        }
        return total;
    }
    public override long Seek(long offset,SeekOrigin origin) { Position=(origin==SeekOrigin.Begin?0:origin==SeekOrigin.Current?Position:length)+offset; return Position; }
    public override void Flush() { }
    public override void SetLength(long value) { throw new NotSupportedException(); }
    public override void Write(byte[] buffer,int offset,int count) { throw new NotSupportedException(); }
    protected override void Dispose(bool disposing) { if(disposing) foreach(var file in files) if(file!=null) file.Dispose(); base.Dispose(disposing); }
}

internal static class Program {
    private const string PayloadHash="__PAYLOAD_SHA256__";
    private const string PackageId="__PACKAGE_ID__";
    private const int PartCount=__PART_COUNT__;
    private static Stream OpenPayload() {
        string parts=Path.Combine(Path.GetDirectoryName(Application.ExecutablePath),"desktop/runtime");
        if(Directory.Exists(parts)) return new ArchiveParts(parts,PartCount);
        var file=File.OpenRead(Application.ExecutablePath);
        if(file.Length>=24) {
            file.Seek(-24,SeekOrigin.End);
            var reader=new BinaryReader(file);
            long offset=reader.ReadInt64(),count=reader.ReadInt64();
            if(Encoding.ASCII.GetString(reader.ReadBytes(8))=="AIVIDEO1" && offset>=0 && count>=0 && offset+count==file.Length-24) return new ArchiveSlice(file,offset,count);
        }
        file.Dispose();
        throw new IOException("실행 환경 폴더가 빠져 있습니다. GitHub에서 AI_video 폴더 전체를 받은 뒤 실행.exe를 눌러 주세요.");
    }
    private static string CacheRoot() {
        string custom=Environment.GetEnvironmentVariable("AIVIDEO_CACHE_ROOT");
        return Path.Combine(String.IsNullOrEmpty(custom)?Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"AI-Video-Classifier"):custom,PackageId);
    }
    private static string Quote(string value) {
        // Arguments here are file paths, without trailing directory separators.
        if(value.IndexOf('"')>=0 || value.IndexOf('\r')>=0 || value.IndexOf('\n')>=0) throw new ArgumentException("Invalid argument");
        return "\""+value+"\"";
    }
    private static string Prepare(Action<int,string> progress) {
        string root=CacheRoot(), ready=Path.Combine(root,".ready");
        using(var mutex=new Mutex(false,"Local\\AIVideo_"+PackageId)) {
            if(!mutex.WaitOne(TimeSpan.FromMinutes(10))) throw new IOException("다른 실행 창의 준비가 끝난 뒤 다시 실행해 주세요.");
            try {
                if(File.Exists(ready) && File.ReadAllText(ready)==PayloadHash && File.Exists(Path.Combine(root,"runtime/python/pythonw.exe"))) return root;
                Directory.CreateDirectory(root);
                using(var slice=OpenPayload()) {
                    progress(0,"처음 실행에 필요한 파일을 확인하고 있습니다…");
                    using(var sha=SHA256.Create()) {
                        string actual=BitConverter.ToString(sha.ComputeHash(slice)).Replace("-","").ToLowerInvariant();
                        if(actual!=PayloadHash) throw new IOException("실행 파일 검증에 실패했습니다. 다시 내려받아 주세요.");
                    }
                    slice.Position=0;
                    using(var zip=new ZipArchive(slice,ZipArchiveMode.Read,true)) {
                        int complete=0;
                        foreach(var entry in zip.Entries) {
                            string destination=Path.GetFullPath(Path.Combine(root,entry.FullName.Replace('/',Path.DirectorySeparatorChar)));
                            if(!destination.StartsWith(Path.GetFullPath(root)+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase)) throw new IOException("Invalid archive path");
                            if(entry.Name.Length>0) {
                                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                                using(var input=entry.Open()) using(var output=File.Create(destination)) input.CopyTo(output);
                            }
                            complete++;
                            if(complete%100==0 || complete==zip.Entries.Count) progress(complete*100/zip.Entries.Count,"분석 프로그램 준비 중… 처음 한 번만 필요합니다.");
                        }
                    }
                }
                File.WriteAllText(ready,PayloadHash);
                return root;
            } finally { mutex.ReleaseMutex(); }
        }
    }
    private static Process Start(string root,string[] args) {
        bool diagnostic=args.Length>=2 && args[0]=="--self-test";
        string python=Path.Combine(root,diagnostic?"runtime/python/python.exe":"runtime/python/pythonw.exe");
        var info=new ProcessStartInfo(python,"-I "+Quote(Path.Combine(root,"desktop/launch.py")));
        foreach(string arg in args) info.Arguments+=" "+Quote(arg);
        info.WorkingDirectory=root;
        info.UseShellExecute=false;
        info.CreateNoWindow=true;
        info.EnvironmentVariables.Remove("PYTHONPATH");
        info.EnvironmentVariables.Remove("PYTHONHOME");
        info.EnvironmentVariables["PYTHONNOUSERSITE"]="1";
        info.EnvironmentVariables["PYTHONUTF8"]="1";
        info.EnvironmentVariables["PYTHONIOENCODING"]="utf-8";
        return Process.Start(info);
    }
    [STAThread]
    private static int Main(string[] args) {
        if(args.Length>=2 && args[0]=="--self-test") {
            try { string root=Prepare(delegate(int n,string m){}); using(var process=Start(root,args)) { process.WaitForExit(); return process.ExitCode; } }
            catch(Exception error) { File.WriteAllText(args[1],error.ToString()); return 1; }
        }
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        var form=new Form { Text="AI 영상 판별", ClientSize=new Size(490,125), StartPosition=FormStartPosition.CenterScreen, FormBorderStyle=FormBorderStyle.FixedDialog, MaximizeBox=false };
        var label=new Label { Left=20, Top=20, Width=450, Height=45, Text="AI 영상 판별 프로그램을 열고 있습니다…" };
        var bar=new ProgressBar { Left=20, Top=80, Width=450, Height=20 };
        form.Controls.Add(label); form.Controls.Add(bar);
        form.Shown+=delegate {
            Task.Run(delegate {
                try {
                    string root=Prepare(delegate(int value,string message) { if(!form.IsDisposed) form.BeginInvoke((Action)delegate { label.Text=message; bar.Value=value; }); });
                    Start(root,args);
                    form.BeginInvoke((Action)delegate { form.Close(); });
                } catch(Exception error) { form.BeginInvoke((Action)delegate { MessageBox.Show(form,error.Message,"실행 실패",MessageBoxButtons.OK,MessageBoxIcon.Error); form.Close(); }); }
            });
        };
        Application.Run(form);
        return 0;
    }
}
